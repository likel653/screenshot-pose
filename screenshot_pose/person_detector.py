"""Honor the requested person confidence for YOLOX graphs with built-in NMS."""
def make_detector(*args, **kwargs):
    from rtmlib import YOLOX
    import numpy as np

    class PersonDetector(YOLOX):
        def postprocess(self, outputs, ratio=1.):
            # RTMLib's built-in-NMS path hardcodes .3 and ignores score_thr.
            # Small low-score background boxes then look like extra people.
            if outputs.shape[-1] == 5:
                rows = outputs[0]
                keep = rows[:, 4] > max(.3, self.score_thr)
                return np.array(rows[keep, :4] / ratio, copy=True)
            return super().postprocess(outputs, ratio)

    return PersonDetector(*args, **kwargs)
