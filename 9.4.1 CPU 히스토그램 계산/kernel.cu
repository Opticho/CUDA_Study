#include "book.h"

#include <time.h>

#define SIZE (100*1024*1024)

int main(void) {
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);

	clock_t start, end;

	start = clock();

	unsigned int histo[256];
	for (int i = 0; i < 256; ++i) {
		histo[i] = 0;
	}

	for (int i = 0; i < SIZE; ++i) {
		++histo[buffer[i]];
	}

	long histoCount = 0;
	for (int i = 0; i < 256; ++i) {
		histoCount += histo[i];
	}

	end = clock();

	printf("histogram Sum: %ld\n", histoCount);
	printf("elapsedTime: %ldms\n", end - start);

	free(buffer);
	return 0;
}