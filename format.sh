#!/usr/bin/env bash
set -ex

pyink --pyink-use-majority-quotes --pyink-indentation 2 --extend-exclude third-party/ .
