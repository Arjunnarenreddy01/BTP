import pandas as pd
import numpy as np

# Dummy grades: rows are students, columns are courses
# values are grades on a 4.0 scale (None means not taken)
grades_dict = {
    "student_id": [1, 2, 3, 4],
    "Math": [3.7, 3.0, 2.5, None],
    "Physics": [3.5, None, 3.8, 2.0],
    "Chemistry": [None, 2.8, 3.2, 3.0],
    "History": [3.0, 3.5, None, 3.2],
}
grades_df = pd.DataFrame(grades_dict).set_index("student_id")

# Dummy professor feedback: each student rated professors on a 5-point scale
prof_feedback_dict = {
    "student_id": [1, 2, 3, 4],
    "Prof_A": [4.5, 3.0, 4.0, 2.5],
    "Prof_B": [3.0, 4.0, 2.5, 3.5],
    "Prof_C": [4.0, 2.5, 3.5, 4.0],
}
prof_feedback_df = pd.DataFrame(prof_feedback_dict).set_index("student_id")

# Additional metadata for courses and professors (optional)
courses = ["Math", "Physics", "Chemistry", "History", "Art", "Music"]
professors = ["Prof_A", "Prof_B", "Prof_C", "Prof_D"]

# map professors to the courses they actually teach (for filtering later)
prof_course_map = {
    "Prof_A": ["Math", "Physics"],
    "Prof_B": ["Chemistry", "History"],
    "Prof_C": ["Math", "History", "Art"],
    # Prof_D currently teaches no core courses in the dummy set
}

# You can also add textual descriptions or latent features here
