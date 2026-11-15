import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import diagnostic_report

class ReportTests(unittest.TestCase):
 def test_offline_payload_round_trip_and_script_escape(self):
  data=dict(body=None,hands={},comparison={'source':'</script><script>bad</script> 中文'})
  page=diagnostic_report.page(data,'source.png')
  serialized=page.split('const data=',1)[1].split(';',1)[0]
  self.assertEqual(json.loads(serialized),data)
  self.assertNotIn('</script>',serialized)
  self.assertNotIn('https://',page)
 def test_image_attribute_escaped(self):
  page=diagnostic_report.page(dict(body=None,hands={}),image='x" onerror="bad')
  self.assertIn('x&quot; onerror=&quot;bad',page)

if __name__=='__main__':unittest.main()
