# PPTX QA Report

- PPTX: `C:\Users\26409\Desktop\Urban Renovation\doc\experiment_archives\urban_binary_policy_v2_20260429\progress_report_20260506\pptx\城市更新二分类实验进展汇报_20260506.pptx`
- Slide count: 10
- Required text missing: None
- File opens as zip: True

## Visual object counts

| Slide | Non-text objects | Native charts | Total shapes |
|---:|---:|---:|---:|
| 1 | 2 | 0 | 17 |
| 2 | 4 | 0 | 18 |
| 3 | 7 | 0 | 28 |
| 4 | 16 | 0 | 37 |
| 5 | 10 | 0 | 28 |
| 6 | 1 | 1 | 10 |
| 7 | 2 | 1 | 10 |
| 8 | 17 | 0 | 40 |
| 9 | 4 | 0 | 21 |
| 10 | 1 | 0 | 7 |


## Render attempt

- Command: `.\.venv-bertopic313\Scripts\python.exe C:\Users\26409\.codex\skills\pptx\scripts\thumbnail.py <pptx> <preview_prefix> --cols 5`
- Result: render preview was blocked in the local Windows environment.
- Error after dependency install: `module 'socket' has no attribute 'AF_UNIX'`.
- Additional environment check: `soffice` was not available on PATH, so true PowerPoint/LibreOffice-rendered PNG previews could not be produced in this session.
- Completed fallback QA: PPTX package opens as zip, slide count is 10, required text was present, placeholder scan was clean, and every slide contains non-text visual objects. Slides 6 and 7 contain native editable charts.
