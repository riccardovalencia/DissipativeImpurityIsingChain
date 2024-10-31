# 1. Put this file in the same folder as your 'driver' code 
#    (the code containing the 'main' function).

# 2. Edit LIBRARY_DIR to point at the location of your ITensor Library
#    source folder (this is the folder that has options.mk in it)
LIBRARY_DIR=/home/ricval/Documenti/itensor
# 3. If your 'main' function is in a file called 'myappname.cc', then
#    set APP to 'myappname'. Running 'make' will compile the app.
#    Running 'make debug' will make a program called 'myappname-g'
#    which includes debugging symbols and can be used in gdb (Gnu debugger);

APP=DMRG_Ising_up_NNN
# 4. Add any headers your program depends on here. The make program
#    will auto-detect if these headers have changed and recompile your app.


# 5. For any additional .cc files making up your project,
#    add their full filenames here.
# CCFILES=$(APP).cpp ${HEADERS_DIR}/external_file.cc ${HEADERS_DIR}/get_data.cc ${HEADERS_DIR}/observables.cc ${HEADERS_DIR}/perform_dmrg.cc ${HEADERS_DIR}/perform_dmrg_meanfield.cc ${HEADERS_DIR}/perform_dmrg_variance.cc ${HEADERS_DIR}/search_ground_state_max_bond_chi.cc ${HEADERS_DIR}/scalar_product_two_cutoffs.cc ${HEADERS_DIR}/scalar_product_different_n0.cc ${HEADERS_DIR}/build_hamiltonian.cc ${HEADERS_DIR}/TEBD.cc ${HEADERS_DIR}/initial_state.cc 
CCFILES=$(APP).cpp 
#################################################################
#################################################################
#################################################################
#################################################################


include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

TENSOR_HEADERS=$(LIBRARY_DIR)/itensor/core.h

#Mappings --------------
OBJECTS=$(patsubst %.cc,%.o, $(CCFILES))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Rules ------------------

%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<
.debug_objs/%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP) $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS)

clean:
	rm -fr .debug_objs *.o $(APP) $(APP)-g

mkdebugdir:
	mkdir -p .debug_objs

