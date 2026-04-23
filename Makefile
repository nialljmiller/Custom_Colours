FC      ?= gfortran
F2PY    ?= f2py
FFLAGS  := -O2 -fPIC -Wall

# Output extension lands in the Python package directory
OUTDIR  := custom_colours

.PHONY: all clean

all: $(OUTDIR)/cc_api$(shell python3-config --extension-suffix 2>/dev/null || echo .so)

$(OUTDIR)/cc_api$(shell python3-config --extension-suffix 2>/dev/null || echo .so): \
		fortran/cc_kernels.f90 fortran/cc_api.f90
	$(F2PY) -c \
		--fcompiler=$(FC) \
		--f90flags="$(FFLAGS)" \
		-m cc_api \
		fortran/cc_kernels.f90 \
		fortran/cc_api.f90 \
		--build-dir /tmp/cc_api_build
	mv cc_api*.so $(OUTDIR)/ 2>/dev/null || true
	mv cc_api*.pyd $(OUTDIR)/ 2>/dev/null || true

clean:
	rm -f $(OUTDIR)/cc_api*.so $(OUTDIR)/cc_api*.pyd
	rm -rf /tmp/cc_api_build
