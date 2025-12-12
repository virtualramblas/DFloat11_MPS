# Makefile for DFloat11 Metal Swift Extension

SWIFT_SOURCE = DFloat11MetalBridge.swift
OUTPUT_LIB = libdfloat11metal.dylib
SWIFT_FLAGS = -emit-library -module-name DFloat11Metal

# Detect SDK path
SDK_PATH := $(shell xcrun --show-sdk-path)

.PHONY: all clean test install

all: $(OUTPUT_LIB)

$(OUTPUT_LIB): $(SWIFT_SOURCE)
	@echo "Building Swift Metal extension..."
	@echo "Using SDK: $(SDK_PATH)"
	swiftc $(SWIFT_FLAGS) \
		-o $(OUTPUT_LIB) \
		-sdk $(SDK_PATH) \
		-target arm64-apple-macosx15.0 \
		$(SWIFT_SOURCE) \
		-framework Metal \
		-framework Foundation
	@echo "Build successful: $(OUTPUT_LIB)"

clean:
	rm -f $(OUTPUT_LIB)
	rm -rf build/
	rm -rf *.dylib
	rm -rf __pycache__
	rm -rf *.swiftmodule *.swiftdoc

test: $(OUTPUT_LIB)
	@echo "Running test..."
	python3 -c "from dfloat11_metal_wrapper import MetalDecoder; d = MetalDecoder(); print('✓ Decoder created successfully')"

install: $(OUTPUT_LIB)
	@echo "Installing library..."
	mkdir -p ~/.local/lib
	cp $(OUTPUT_LIB) ~/.local/lib/
	@echo "Library installed to ~/.local/lib/"

.DEFAULT_GOAL := all