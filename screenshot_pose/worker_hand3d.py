"""Hand-only model jobs and honest A/B previews; never alter body observations."""
import copy
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
try:
    from . import hand3d, worker, pose_data, hand_geometry as hg, multi_view
except ImportError:
    import hand3d, worker, pose_data, hand_geometry as hg, multi_view


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview(base, candidates, reports, path):
    from PIL import ImageDraw
    sections = []
    for backend, data in candidates.items():
        im = worker.prepare_image(base['image']['path'])
        draw = ImageDraw.Draw(im)
        details = []
        for side, info in reports[backend].items():
            if info['adopted']:
                hd = data['orientation']['hands'][side]
                xy = np.asarray(hd.get('reference_landmarks', hd.get('landmarks'))) * im.size
                # Only primary-view pixels belong on this image. Auxiliary
                # predictions keep their own reference and appear in JSON.
                same_view = Path(hd.get('reference_image', base['image']['path'])).resolve() == Path(base['image']['path']).resolve()
                if same_view:
                    color = '#69e4ff' if side == 'L' else '#ffb577'
                    for a, b in hand3d.EDGES:
                        draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=3)
                    for i, point in enumerate(xy):
                        x, y = point
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
                        draw.text((x+4, y), str(i), fill=color)
                details.append(side + '：已得到 21 个三维点（几何检查通过，不代表准确率已验证）')
            else:
                details.append(side + '：未采用；' + info['reason'])
        image_path = path.with_name(path.stem + '_' + backend.lower() + '.png')
        worker.save_png(im, image_path)
        label={'HAMER':'原版 HaMeR','HAMER_TEXTURE':'纹理监督增强 HaMeR','WILOR':'WiLoR'}.get(backend,backend)
        sections.append('<section><h2>' + label + '</h2><img src="' + image_path.name + '"><p>' +
                        '<br>'.join(html.escape(v) for v in details) + '</p></section>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>三维手部模型对比</title>'
        '<style>body{background:#151b28;color:#e9efff;font:16px system-ui;margin:32px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}'
        'section{background:#222c40;padding:20px;border-radius:16px}img{width:100%}p{line-height:1.8}</style>'
        '<h1>同图手部三维重建对比</h1><p>左右均指人物自身；对比不会自动选择赢家或应用角色。'
        '未采用的手保留原始数据，不作为该模型识别成功。投影一致不等于深度正确。</p><main>' +
        ''.join(sections) + '</main>', encoding='utf8')


def run(input, request, model, output, native_root_path=None, session=None):
    try:
        from . import worker_hands as legacy
    except ImportError:
        import worker_hands as legacy
    source, req, out = map(lambda v: Path(v).resolve(), (input, request, output))
    if req.stat().st_size > 65536:
        raise ValueError('手部请求过大')
    selections = json.loads(req.read_text(encoding='utf8'))
    if not selections or any(s not in ('L', 'R') for s in selections) or any(not isinstance(v, dict) for v in selections.values()):
        raise ValueError('无效手部请求')
    backends = {v.get('backend', 'LOCAL') for v in selections.values()}
    if len(backends) != 1 or not backends <= set(hand3d.KNOWN_BACKENDS)|{'COMPARE','COMPARE_TEXTURE'}:
        raise ValueError('请选择一致的三维手部模型')
    if any(v.get('gesture', 'NONE') not in ('NONE', 'V', 'GRIP', 'OPEN', 'FIST') for v in selections.values()):
        raise ValueError('手部参数无效')
    choice = next(iter(backends))
    comparing=choice in ('COMPARE','COMPARE_TEXTURE')
    base = pose_data.read_pose(source)
    source_hash = digest(source)
    runtime = Path(model).resolve().parent
    outputs = [out, out.with_suffix('.report.json'), out.with_suffix('.html')]
    outputs += [out.with_name(out.stem + '.' + b.lower() + '.json') for b in hand3d.KNOWN_BACKENDS]
    outputs += [out.with_name(out.stem + '.' + b.lower() + '.report.json') for b in hand3d.KNOWN_BACKENDS]
    outputs += [out.with_name(out.stem + '_' + b.lower() + '.png') for b in hand3d.KNOWN_BACKENDS]
    protected = {source, req, Path(model).resolve(), Path(base['image']['path']).resolve()}
    protected |= {Path(v['image']).resolve() for v in selections.values() if v.get('image')}
    # A job owns one fresh output folder, so reports/previews cannot replace
    # source images, checkpoint files or unrelated user files.
    if len(set(outputs)) != len(outputs) or set(outputs) & protected or any(p.exists() for p in outputs):
        raise ValueError('手部输出不能覆盖输入或已有结果')
    if out.parent == runtime or runtime / 'hand3d' == out.parent or (runtime / 'hand3d') in out.parents:
        raise ValueError('手部结果不能写入模型目录')
    out.parent.mkdir(parents=True, exist_ok=True)
    references, contexts, context_errors = {}, {}, {}
    for side, selection in selections.items():
        try:
            image = selection.get('image')
            d, rotation = base, np.eye(3)
            if image:
                same = legacy.mv.pixels(image) == legacy.mv.pixels(base['image']['path'])
                person = int(selection.get('person_index', 1)) - 1
                if same and person != base.get('person_index', 0):
                    raise ValueError('相同参考图的人物序号与主图不一致')
                if not same:
                    key = (image, person)
                    if key not in references:
                        references[key], _ = legacy.reference(image, person, model, .5, native_root_path or legacy.worker_sam.native_root())
                    d = references[key]
                    rotation = np.asarray(multi_view.align(base, d)['rotation'])
            contexts[side] = (d, rotation)
        except Exception as exc:
            context_errors[side] = str(exc)
    started = time.perf_counter()
    candidates, reports, paths = {}, {}, {}
    requested = hand3d.BACKENDS if choice == 'COMPARE' else ('HAMER','HAMER_TEXTURE') if choice=='COMPARE_TEXTURE' else (choice,)
    own = session is None
    try:
        for backend in requested:
            result, info = copy.deepcopy(base), {}
            state = hand3d.readiness(runtime, backend)
            for side, selection in selections.items():
                meta = dict(reference=selection.get('image') or base['image']['path'], gesture=selection.get('gesture', 'NONE'), backend=backend)
                try:
                    if side in context_errors:
                        raise ValueError(context_errors[side])
                    if not state['ready']:
                        raise ValueError(backend + ' 缺少：' + '、'.join(state['missing']))
                    if session is None:
                        session = hand3d.Session(runtime,screenshot_precision=selection.get('precision',True))
                    d, rotation = contexts[side]
                    if own:session.screenshot_precision=selection.get('precision',True)
                    session.original_image=d['image'].get('original_path')
                    refined, report = hand3d.refine(d, backend, session, sides=(side,))
                    if not report[side]['adopted']:
                        raise ValueError(report[side]['reason'])
                    hd = hg.transport(refined['orientation']['hands'][side], rotation)
                    hd.update(reference_image=d['image']['path'], gesture_constraint=selection.get('gesture', 'NONE'))
                    result.setdefault('orientation', {}).setdefault('hands', {})[side] = hd
                    meta.update(report[side])
                except Exception as exc:
                    meta.update(adopted=False, reason=str(exc)[:600])
                info[side] = meta
            result['hand_refinement'] = dict(primary_json=str(source), primary_sha256=source_hash, primary_image=base['image']['path'],
                hands=info, body_preserved=True, auto_applied=False, backend=backend)
            candidate = out.with_name(out.stem + '.' + backend.lower() + '.json')
            worker.write_json(candidate, pose_data.validate(result))
            worker.write_json(candidate.with_suffix('.report.json'), result['hand_refinement'])
            paths[backend] = str(candidate)
            candidates[backend], reports[backend] = result, info
    finally:
        if own and session:
            session.close()
    result = copy.deepcopy(base) if comparing else copy.deepcopy(candidates[choice])
    report = dict(primary_json=str(source), primary_sha256=source_hash, primary_image=base['image']['path'],
        body_preserved=True, auto_applied=False, backend=choice, candidates=paths, candidate_reports=reports,
        seconds=time.perf_counter()-started,
        hands={s: dict(reference=v.get('image') or base['image']['path'], gesture=v.get('gesture', 'NONE'),
                       adopted=False, reason='双模型结果已分别保存，等待选择') for s,v in selections.items()}
              if comparing else reports[choice])
    page = out.with_suffix('.html')
    preview(base, candidates, reports, page)
    report['comparison_html'] = str(page)
    result['hand_refinement'] = report
    worker.write_json(out, pose_data.validate(result))
    worker.write_json(out.with_suffix('.report.json'), report)
    print(json.dumps(dict(ok=True, adopted=sum(v['adopted'] for v in report['hands'].values()), comparison=str(page))), flush=True)
    return result
