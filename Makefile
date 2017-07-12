CC=gcc
CFLAGS=-g -Wall
LFLAGS= -lgsl -lgslcblas -lm -lpthread

all: FAST

FAST:  GWiS.o hashtable.o Logistic.o BFLogistic_newton.o BFLogistic_fr.o QuickSort.o 
		$(CC)  -Wall -o FAST GWiS.o hashtable.o Logistic.o BFLogistic_newton.o BFLogistic_fr.o QuickSort.o $(LFLAGS) 
		rm *.o
GWiS.o: ./Code/GWiS.c
	$(CC) $(CFLAGS) -c ./Code/GWiS.c -o GWiS.o
Logistic.o: ./Code/Logistic.c
	$(CC) $(CFLAGS) -c ./Code/Logistic.c -o Logistic.o
BFLogistic_newton.o: ./Code/BFLogistic_newton.c
	$(CC) $(CFLAGS) -c ./Code/BFLogistic_newton.c -o BFLogistic_newton.o
BFLogistic_fr.o: ./Code/BFLogistic_fr.c
	$(CC) $(CFLAGS) -c ./Code/BFLogistic_fr.c -o BFLogistic_fr.o
hashtable.o: ./Code/hashtable.c
	$(CC) $(CFLAGS) -c ./Code/hashtable.c -o hashtable.o 
QuickSort.o: ./Code/QuickSort.c 
	$(CC) $(CFLAGS) -c ./Code/QuickSort.c -o QuickSort.o 
	
clean:
	rm FAST
	rm *.o
