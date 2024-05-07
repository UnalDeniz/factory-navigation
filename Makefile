VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate build
	$(PYTHON) python/main.py


$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

build: cpp/dynamic_window.cpp cpp/sobol.cpp cpp/sobol.hpp
	g++ -shared -o build/libdwa.so cpp/dynamic_window.cpp cpp/sobol.cpp cpp/sobol.hpp -fPIC


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf build
