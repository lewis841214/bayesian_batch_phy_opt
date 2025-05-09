python tests/test_algorithms/unified_benchmark.py --compare --algorithms nn-qnehvi --problem nonlinear --budget 100 --batch-size 10 --output-dir output/nn_nonlinear

python tests/test_algorithms/unified_benchmark.py --compare --algorithms qnehvi nn-qnehvi --problem complex_categorical --budget 100 --batch-size 10 --output-dir output/nn_complex_categorical


python tests/test_algorithms/unified_benchmark.py --compare --algorithms qnehvi nn-qnehvi --problem complex_categorical --budget 100 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map


python tests/test_algorithms/unified_benchmark.py --compare --algorithms qnehvi nn-qnehvi --problem complex_categorical --budget 100 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map

python tests/test_algorithms/unified_benchmark.py --compare --algorithms randomsearch nsga2 nsga3 moead  qnehvi nn-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map_9_3_200
python tests/test_algorithms/unified_benchmark.py --compare --algorithms  nn-qnehvi --problem complex_categorical --budget 100 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map_test_o_transform


python tests/test_algorithms/unified_benchmark.py --compare --algorithms randomsearch qnehvi nn-qnehvi --problem nonlinear --budget 100 --batch-size 10 --output-dir output/nonlinear_hidden_map

python tests/test_algorithms/unified_benchmark.py --compare --algorithms randomsearch nsga2 nsga3 moead  qnehvi nn-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map_9_3_200

python tests/test_algorithms/unified_benchmark.py --compare --algorithms randomsearch nsga2 nsga3 moead  qnehvi nn-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/nn_complex_categorical_hidden_map_8_3_200

python tests/test_algorithms/unified_benchmark.py --compare --algorithms  qnehvi nnk-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/nnK_6_3

 python tests/test_algorithms/unified_benchmark.py --compare --algorithms  randomsearch nsga2 nsga3 moead qnehvi nnk-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/nnK_8_2_1lay_30_seed_47

python tests/test_algorithms/unified_benchmark.py --compare --algorithms   nnk-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/check_test_space

python tests/test_algorithms/unified_benchmark.py --compare --algorithms    randomsearch nsga2 nsga3 moead  qnehvi nnk-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/test_hybrid_8_2

python tests/test_algorithms/unified_benchmark.py --compare --algorithms    randomsearch nsga2 nsga3 moead  qnehvi nnk-qnehvi --problem complex_categorical --budget 200 --batch-size 10 --output-dir output/test_full_7_2_47


# randomsearch nsga2 nsga3 moead  qnehvi nn-qnehvi

mixed': MixedParameterTestProblem(),
    'nonlinear': NonlinearTestProblem(),
    'discrete': DiscreteTestProblem(),
    'constrained': ConstrainedTestProblem(),
    'large_mixed': LargeMixedParameterTestProblem(),
    'category_matrix': CategoryMatrixTestProblem(),
    'complex_categorical': ComplexCategoryEmbeddingProblem(),
    'complex_interaction': ComplexInteractionProblem()