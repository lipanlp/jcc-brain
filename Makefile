PY := .venv/bin/python

.PHONY: setup test gui web sync app clean

setup:            ## 创建虚拟环境并安装依赖
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e '.[dev,build]'

test:             ## 运行单元测试
	$(PY) -m pytest -q

gui: web          ## 打开图形界面
	$(PY) -m jccbrain gui

web:              ## 构建图形界面
	cd web && npm ci && npm run build

sync:             ## 从 dataj.cc 同步最新数据
	$(PY) -m jccbrain db sync

app:              ## 打包 dist/金铲铲大脑.app
	./scripts/build_app.sh

clean:
	rm -rf build dist .pytest_cache
