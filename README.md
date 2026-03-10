# LordTorch

A lightweight C++ project for building a Torch-like tensor and neural network framework from scratch.

## Overview

**LordTorch** is a learning-oriented C++ project that aims to build a simplified deep learning framework inspired by Torch and PyTorch.

This project is not intended to replace PyTorch or compete with existing production-grade frameworks. Instead, its purpose is to explore and implement the core ideas behind a modern tensor library and neural network runtime in a clean, structured, and understandable way.

LordTorch focuses on re-creating the fundamental building blocks of a Torch-like system in pure C++, including tensor storage, shape and stride management, elementwise operations, reductions, matrix multiplication, neural network modules, optimizers, and eventually automatic differentiation.

In short, LordTorch is an educational mini-framework for understanding how a Torch-like system can be built from the ground up.

## Motivation

PyTorch is one of the most popular deep learning frameworks because it provides:

- an intuitive tensor API
- flexible dynamic computation
- practical neural network abstractions
- convenient optimization workflows

However, many of its core mechanisms are hidden inside a large, complex, and highly optimized codebase.

LordTorch exists to make these ideas easier to study.

By implementing a small Torch-like framework in C++, this project helps clarify how the following concepts work internally:

- tensor data structures
- storage, shapes, strides, and views
- tensor creation and manipulation
- elementwise mathematical operators
- reduction operators such as sum and mean
- matrix multiplication
- module and parameter abstractions
- loss functions and optimizers
- the foundations of autograd and backpropagation

This project is especially useful for students, researchers, and developers who want to understand deep learning frameworks beyond high-level API usage.

## Project Goal

The long-term goal of LordTorch is to build a compact, modular, and educational C++ framework that resembles a minimal version of Torch / PyTorch in overall design philosophy.

The framework is intended to provide:

- a core `Tensor` abstraction
- reusable storage and implementation layers
- common tensor creation and transformation APIs
- arithmetic, reduction, and matrix operations
- neural network building blocks such as `Module`, `Parameter`, and `Linear`
- optimizer utilities such as `SGD`
- a clear extension path toward automatic differentiation and trainable models

The emphasis of this project is on:

- clarity
- modularity
- extensibility
- educational value

rather than production-level performance or full PyTorch compatibility.

## What “Torch-like” Means Here

LordTorch is inspired by the design style of Torch / PyTorch, especially in the following ways:

- tensors are the central data abstraction
- neural network components are organized as modules
- trainable values are represented as parameters
- optimizers update collections of parameters
- the system is designed to evolve toward dynamic autograd

This repository does **not** attempt to reproduce the entire PyTorch feature set. Instead, it focuses on implementing a minimal and understandable subset of the ideas that make Torch-like frameworks expressive and powerful.

## Current Features

At its current stage, LordTorch already provides the basic skeleton of a C++ mini deep learning framework.

Implemented or scaffolded components include:

- tensor storage abstraction
- tensor implementation layer
- tensor creation utilities
- basic elementwise operations
- reduction operations such as `sum` and `mean`
- matrix multiplication interface
- activation-related operator structure
- module base class
- parameter abstraction
- linear layer skeleton
- loss and optimizer skeletons
- CMake-based build system
- example programs
- unit tests for tensor basics and reductions

## Current Status

LordTorch is currently in an early but functional stage.

The project already includes:

- a compilable C++ project structure
- a working CMake-based build workflow
- successful builds for the library, examples, and tests
- passing unit tests for tensor basics and reduction behavior

At the same time, several major parts are still under active development, including:

- richer tensor view and manipulation support
- dimension-wise reductions
- more complete neural network components
- full automatic differentiation
- backward propagation for core operators
- end-to-end training loops
- broader operator coverage

## Intended Audience

LordTorch is designed for people who want to understand how frameworks like Torch and PyTorch are built internally.

It may be useful for:

- students learning deep learning systems
- researchers interested in tensor framework design
- C++ developers exploring numerical computing libraries
- anyone who wants to understand tensors, autograd, and neural network abstractions from first principles

## Project Structure

A simplified overview of the repository structure is shown below:

```text
LordTorch/
├── include/mtorch/        # Public headers
├── src/                   # Core implementation
├── tests/                 # Unit tests
├── examples/              # Example programs
├── docs/                  # Design notes and architecture docs
└── CMakeLists.txt         # Build configuration
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run examples

```bash
./build/example_tensor_basics
./build/example_linear_regression
```

## Roadmap


1. Planned future improvements include:
2. better tensor indexing and view semantics
3. sum(dim) and mean(dim) support
4. more activation and manipulation operators
5. automatic differentiation engine
6. backward support for core tensor operations
7. improved Module and Sequential APIs
8. more complete optimizers
9. end-to-end training examples
10. optional Python bindings in the future
