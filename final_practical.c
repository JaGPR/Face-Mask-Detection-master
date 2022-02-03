#include <stdio.h>
#include <stdlib.h>
struct type {
  int date;
  char name[30];
  int salary;
};

int main() {
  struct type *ptr;
  int noOfRecords;
  printf("Enter the number of records: ");
  scanf("%d", &noOfRecords);
  ptr = (struct type *)malloc(noOfRecords * sizeof(struct type));
  for (int i = 0; i < noOfRecords; ++i) {
    printf("Enter eid ,name and salary:\n");
    scanf("%d%s%d",&(ptr + i)->date, &(ptr + i)->name, &(ptr + i)->salary);
  }

  printf("Displaying Information:\n");
  for (int i = 0; i < noOfRecords; ++i) {
    printf("%d\t%s\t%d\n",(ptr + i) -> date, (ptr + i)->name, (ptr + i)->salary);
  }

  free(ptr);

  return 0;
}