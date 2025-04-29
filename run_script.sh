python tests/test_algorithms/unified_benchmark.py --compare --algorithms nn-qnehvi --problem nonlinear --budget 100 --batch-size 10 --output-dir output/nn_nonlinear

python tests/test_algorithms/unified_benchmark.py --compare --algorithms qnehvi nn-qnehvi --problem complex_categorical --budget 100 --batch-size 10 --output-dir output/nn_complex_categorical