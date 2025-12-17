CXX      := g++
CXXFLAGS := -O3 -std=gnu++20 -g
LDFLAGS  := -fopenmp -pthread -luring -laio
INCLUDES := -I ./include
CACHE_FACTORY := ./include/cache/CacheFactory.cpp 
# TARGETS := ./experiments/rmi_falcon_test  

falcon_wait_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/parallel5.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_eps_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/parallel2.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_batch_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/parallel4.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_range_eps_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/range_parallel.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_range_single_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/falcon_range_single.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_point_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/falcon_point.cpp $(CACHE_FACTORY) $(LDFLAGS)

falcon_join_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/join_parallel.cpp $(CACHE_FACTORY) $(LDFLAGS)
	
rmi_disk_test_books:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./include/rmi/books_rmi.cpp ./experiments/benchmark/rmi_disk_test_books.cpp $(LDFLAGS)

pgm_disk_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/benchmark/pgm_disk_test.cpp $(LDFLAGS)

pgm_mem_range_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/benchmark/pgm_mem_range_test.cpp $(LDFLAGS)

aulid_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./experiments/benchmark/aulid_test.cpp $(LDFLAGS)

bplustree_test:
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ./test ./include/bplustree/stx_disk_kv.cpp ./experiments/benchmark/bplustree_test.cpp $(LDFLAGS)
	g++ -O3 -g -pthread ./include/bplustree/stx_disk_kv.cpp -I ./include ./experiments/benchmark/bplustree_test.cpp -o test $(LDFLAGS)