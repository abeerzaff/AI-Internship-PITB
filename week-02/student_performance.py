
students = []
total_students = int(input("Enter number of students: "))
for i in range(total_students):
    print()
    print(f"Student {i+1}")
    name = input("Enter name: ")
    marks = float(input("Enter marks: "))
    students.append({
        "Name": name,
        "Marks": marks
    })
print()
print("Student Performance Summary")
marks = 0
for student in students:
    marks += student["Marks"]
    if student["Marks"] >= 80:
        grade = "A"
    elif student["Marks"] >= 70:
        grade = "B"
    elif student["Marks"] >= 60:
        grade = "C"
    elif student["Marks"] >= 50:
        grade = "D"
    else:
        grade = "F"
    print(f"{student['Name']} - Marks: {student['Marks']} | Grade: {grade}")
average = marks / total_students
print()
print("Class Average:", round(average, 2))
print("Highest Marks:", max(student["Marks"] for student in students))
print("Lowest Marks:", min(student["Marks"] for student in students))