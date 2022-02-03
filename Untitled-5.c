#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Student {
    
    int roll_number;
    int year;
    char* name;
    char* branch;
    
};
  

int main()
{
    printf("NAME : JAGPREET SINGH\nUID : 21BCS1337\n");
    int i = 0, n = 5;
    struct Student student[5];
    student[0].roll_number = 1;
    student[0].name = "STUDENT 1";
    student[0].year = 2021;
    student[0].branch = "CSE";
  
    student[1].roll_number = 2;
    student[1].name = "Student 2";
    student[1].year = 2021;
    student[1].branch = "BBA";
  
    student[2].roll_number = 3;
    student[2].name = "Studnet 3";
    student[2].year = 2021;
    student[2].branch = "MECHANICAL";
  
    student[3].roll_number = 4;
    student[3].name = "Student4";
    student[3].year = 2021;
    student[3].branch = "CIVIL";
  
    student[4].roll_number = 5;
    student[4].name = "Student5";
    student[4].year = 2021;
    student[4].branch = "BSC";
  
 
    printf("Student Records:\n\n");
    for (i = 0; i < n; i++) {
        printf("\tName = %s\n", student[i].name);
        printf("\tRoll Number = %d\n", student[i].roll_number);
        printf("\tyear = %d\n", student[i].year);
        printf("\tBranch = %s\n\n", student[i].branch);
    }
  
    return 0;
}