import pandas as pd
import random

rows = []
num_feedback = 30000
courses = 120
profs = 80
semesters = ["2023A","2023B","2024A","2024B","2025A"]

for i in range(num_feedback):

    clarity = random.randint(2,10)
    workload = random.randint(1,10)
    interaction = random.randint(2,10)
    attendance = random.randint(1,10)
    assignments = random.randint(2,10)
    organization = random.randint(2,10)

    overall = round((clarity + interaction + assignments + organization)/4)

    rows.append([
        i,
        random.randint(1,courses),
        random.randint(1,profs),
        random.choice(semesters),
        clarity,
        workload,
        interaction,
        attendance,
        assignments,
        overall,
        organization,
        None
    ])

cols = [
"feedback_id","course_id","professor_id","semester",
"rating_clarity","rating_workload","rating_interaction",
"rating_attendance_strictness","rating_assignments",
"overall_rating","course_organization","comment"
]

df = pd.DataFrame(rows, columns=cols)
df.to_csv("data/feedback.csv", index=False)