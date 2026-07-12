# 철근망 카운터 웹 프로그램

shlee 브랜치의 3차시도와 main 브랜치의 4차시도를 하나의 화면에서 같은 입력 이미지에 적용해 비교하는 Flask 프로그램입니다.

## 실행

```bash
python3 -m pip install -r web_requirements.txt
python3 web_app.py
```

가상환경 사용을 권장합니다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r web_requirements.txt
.venv/bin/python web_app.py
```

브라우저에서 `http://127.0.0.1:5000`을 열고 이미지를 업로드합니다.

처음 화면에서 이미지를 한 번 업로드한 뒤 ROI 방법을 선택합니다. 업로드한 이미지는 임시 보관되므로, 두 버튼을 각각 눌러도 이미지를 다시 첨부할 필요가 없습니다.

두 방법론을 모두 실행하면 결과는 화면 전체 폭의 탭 형태로 표시되며, 탭을 눌러 수동 ROI 결과와 자동 ROI 결과를 전환해서 확인할 수 있습니다.

- `수동 ROI`: 이미지 위에서 철근망 외곽 네 점을 마우스로 클릭한 뒤 shlee 브랜치 3차시도를 실행
- `자동 ROI`: main 브랜치 4차시도의 Hough 직선 조합으로 외곽을 자동 검출한 뒤 실행

수동 ROI 화면에서는 점을 좌상단 → 우상단 → 우하단 → 좌하단 순서로 클릭합니다. 클릭한 좌표는 사용자가 직접 입력할 필요 없이 브라우저가 자동으로 계산합니다.

3차시도는 실행 시 `git show shlee:결과/3차 시도/rebar_grid_counter.py`로 shlee 브랜치의 소스를 직접 읽습니다. 따라서 main 브랜치의 동명 파일과 섞이지 않습니다. Git 저장소가 아닌 곳에 복사한 경우에는 `결과/3차 시도/rebar_grid_counter.py`를 fallback으로 사용합니다.
