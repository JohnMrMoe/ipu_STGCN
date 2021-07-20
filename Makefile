
CC 			  := g++ -std=c++11 -g
FLAGS 	  := -lm
IPU			  := -D_IRL_IPU
RDLK			:= -D_QUICK_READ
DRND			:= -D_DERANDOMIZE
VERI      := -D_VERIFICATION
POPLAR	  := -lpoplar -lpoputil -lpopops -lpoplin -lpopsparse -lpopnn -lpoprand -fopenmp
POPVISION := POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./report"}'
# POP :=

C_FILES = $(wildcard util/*.cpp) $(wildcard data/*.cpp) $(wildcard model/*.cpp)
H_FILES = $(wildcard util/*.hpp) $(wildcard data/*.hpp) $(wildcard model/*.hpp)

.PHONY: clean run vrun run_ipu vrun_ipu document redoc import



stgcn: clean $(H_FILES)
	$(CC) main.cpp $(C_FILES) -o $@ $(FLAGS) $(POPLAR) $(DRND) $(VERI)

stgcn_ipu: clean $(H_FILES)
	$(CC) main.cpp $(C_FILES) -o $@ $(FLAGS) $(IPU) $(POPLAR) $(DRND) $(VERI)

run: import
	./stgcn --batch_size 30
vrun: import
	valgrind ./stgcn --batch_size 30
clean_run: clean stgcn run
clean_vrun: clean stgcn vrun

run_ipu: import
	./stgcn_ipu --batch_size 30
vrun_ipu: import
	valgrind ./stgcn_ipu --batch_size 30
clean_ipu_run: clean stgcn_ipu run_ipu
clean_ipu_vrun: clean stgcn_ipu vrun_ipu

import:
	rm -r rawdata
	cp -r ../port_STGCN_\(tf\ Python\)/rawdata .


document:
	./stgcn > doc.txt

redoc: clean stgcn
	valgrind --leak-check=full ./stgcn > doc.txt

clean:
	rm -f stgcn
	rm -f stgcn_ipu