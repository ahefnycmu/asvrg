.PHONY: all clean lib tests run_tests
.SECONDARY: %.cpp %.h %.cc

##################################################################
#Configurations
##################################################################
CONFIG ?= opt
GCC ?= g++

ifeq ($(CONFIG), dbg)
USEOPENMP = 0
DEBUG = 1
else ifeq ($(CONFIG), opt)
USEOPENMP = 1
DEBUG = 0
else 
$(error Invalid configuration)
endif

##################################################################
#Directories
##################################################################
SRCDIR = src
SRC = $(shell find $(SRCDIR)/ -type f -name '*.cpp') 

MAINSRCDIR = src_main
MAINSRC = $(wildcard $(MAINSRCDIR)/*.cpp)
TESTSRCDIR = src_test
TESTSRC = $(wildcard $(TESTSRCDIR)/*.cpp)

OBJDIR = obj/$(CONFIG)
OBJ = $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

LIBDIR = lib/$(CONFIG)
LIBTARGET = $(LIBDIR)/libsvrg.a

BINDIR = bin/$(CONFIG)
BINTARGET = $(MAINSRC:$(MAINSRCDIR)/%.cpp=$(BINDIR)/%)

TESTTARGETDIR = $(BINDIR)
TESTTARGET = $(TESTSRC:$(TESTSRCDIR)/%.cpp=$(TESTTARGETDIR)/%)
TESTSUCCESSFLAG = $(TESTSRC:$(TESTSRCDIR)/%.cpp=$(TESTTARGETDIR)/%.sf)

##################################################################
#Flags
##################################################################
ifeq ($(USEOPENMP),1)
OMPFLAG = -fopenmp -DUSE_OPENMP
OMPLFLAG = -fopenmp
else
OMPFLAG = -Wno-unknown-pragmas
OMPLFLAG = 
endif

ifeq ($(DEBUG),1)
OFLAG = -g -DDEBUG -O0
else
OFLAG = -O3
endif

ifeq ($(USEMPI),1)
CPP = mpic++
else
CPP = $(GCC)
endif

#Add -p if profiling is needed
CFLAG = -rdynamic -Wall -Wno-reorder -I. -I$(SRCDIR) -I./ext/include -MMD -MP -std=c++0x -include common.h $(OFLAG) $(OMPFLAG) -pg
LFLAG = -fprofile-arcs -ftest-coverage -lstdc++ $(OMPLFLAG) $(OFLAG) -pg

all: $(BINTARGET)

tests: $(TESTTARGET)

run_tests: $(TESTSUCCESSFLAG)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(BINDIR):
	mkdir -p $(BINDIR) 

$(OBJDIR)/%.ico : $(ICEOUTDIR)/%.cpp 	
	@mkdir -p "$(@D)"
	$(CPP) $(CFLAG) -I"$(<D)" -c $< -o $@

$(OBJDIR)/%.pbo : $(PBUFOUTDIR)/%.pb.cc			
	@mkdir -p "$(@D)"	
	$(CPP) $(CFLAG) -I"$(<D)" -c $< -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp 	
	@mkdir -p "$(@D)"
	$(CPP) $(CFLAG) -c $< -o $@

-include $(OBJ:%.o=%.d)

$(LIBTARGET): $(OBJ) $(ICEOBJ) $(PBUFOBJ) | $(LIBDIR)
	mkdir -p $(LIBDIR)
	ar crs $(LIBTARGET) $(OBJ) $(ICEOBJ) $(PBUFOBJ)

$(BINDIR)/%: $(MAINSRCDIR)/%.cpp $(LIBTARGET) | $(BINDIR)
	$(CPP) --version
	mkdir -p $(BINDIR)
	$(CPP) $(CFLAG) $< $(LIBTARGET) $(LFLAG) -o $@

include $(wildcard $(BINDIR)/*.d)

$(TESTTARGETDIR)/%: $(TESTSRCDIR)/%.cpp $(LIBTARGET) | $(BINDIR)
	mkdir -p $(BINDIR)
	$(CPP) $(CFLAG) -include $(TESTSRCDIR)/common_test.h $< $(LIBTARGET) $(LFLAG) -o $@

$(TESTTARGETDIR)/%.sf: $(TESTTARGETDIR)/%
	$<
	touch $@

clean:	
	rm -rf $(BINDIR)
	rm -rf $(OBJDIR)
	rm -rf $(LIBDIR)
	rm -rf $(TESTTARGETDIR)

clean_all:
	rm -rf obj
	rm -rf bin
	rm -rf lib

rebuild: clean all
