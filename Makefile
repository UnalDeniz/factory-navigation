VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
ACTIVATE_NVIDIA_PROPRIETY = __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
ACTIVATE_GPU_FOSS = DRI_PRIME=1

run: $(VENV)/bin/activate build
	$(ACTIVATE_NVIDIA_PROPRIETY) $(ACTIVATE_GPU_FOSS) $(PYTHON) python/main.py


$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

build: cpp/dynamic_window.cpp cpp/sobol.cpp cpp/sobol.hpp
	mkdir -p build
	g++ -shared -o build/libdwa.so cpp/dynamic_window.cpp cpp/sobol.cpp cpp/sobol.hpp -fPIC


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf build
