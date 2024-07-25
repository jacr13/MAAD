#!/usr/bin/env bash

rm -r .mypy_cache || true

rm ./*.log || true
rm -r spawn/* || true
rm -r out || true