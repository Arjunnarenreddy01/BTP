import pandas as pd
import numpy as np
import random

# Load course latent factors
course_factors = pd.read_csv("../data/course_latent_factors.csv")

# Grade mapping to numerical scores
GRADE_MAPPING = {
    'A*': 10.0,
    'A': 10.0,
    'A-': 9.0,
    'B': 9.0,
    'B-': 7.0,
    'C': 6.0,
    'C-': 5.0,
    'F': 0.0
}

def create_student_with_random_history(num_prev_courses=8, num_oes=40):
    """
    Create a student with:
    - Random history of previous courses with grades
    - Random list of 40 OEs (Offered Electives) from 120 courses
    """
    all_courses = sorted(course_factors['course_id'].unique())
    
    # Random previous courses taken (student has grades for these)
    prev_courses = random.sample(all_courses, k=num_prev_courses)
    grades = [random.choice(list(GRADE_MAPPING.keys())) for _ in prev_courses]
    
    # Student history: {course_id: grade}
    student_history = {course: grade for course, grade in zip(prev_courses, grades)}
    
    # Random OEs offered to student (40 courses from 120)
    offered_electives = sorted(random.sample(all_courses, k=num_oes))
    
    return {
        'history': student_history,
        'offered_electives': offered_electives
    }

def derive_student_latent_vector(student_data, course_factors, latent_factor_cols):
    """
    Derive student latent vector from their grade history.
    
    For each course taken, we get the grade score and course latent factors.
    We compute a weighted average of course factors weighted by performance.
    
    Args:
        student_data: dict with 'history' (course_id -> grade mapping)
        course_factors: DataFrame with course latent factors
        latent_factor_cols: list of latent factor column names
    
    Returns:
        np.array of student latent vector (same dimension as course factors)
    """
    history = student_data['history']
    
    if not history:
        # No history, return zero vector
        return np.zeros(len(latent_factor_cols))
    
    student_vector = np.zeros(len(latent_factor_cols))
    total_score = 0
    
    for course_id, grade in history.items():
        # Get course latent factors
        course_row = course_factors[course_factors['course_id'] == course_id]
        if len(course_row) == 0:
            continue
            
        # Get performance score for this grade
        performance_score = GRADE_MAPPING[grade]
        
        # Get course latent factors
        course_vector = course_row[latent_factor_cols].values[0]
        
        # Weighted sum: performance * course_factors
        student_vector += performance_score * course_vector
        total_score += performance_score
    
    # Normalize by total score
    if total_score > 0:
        student_vector /= total_score
    
    return student_vector

def student_query(student_num=1):
    """
    Create a student query with:
    - Student's previous course grades
    - List of offered electives (40 courses)
    - Derived student latent vector
    """
    latent_cols = ['latent_computational_math_heavy', 'latent_theory_heavy', 
                   'latent_difficulty_rigor', 'latent_practicality', 'latent_ease']
    
    print(f"\n{'='*70}")
    print(f"STUDENT QUERY #{student_num}")
    print(f"{'='*70}\n")
    
    # Create student
    student = create_student_with_random_history(num_prev_courses=8, num_oes=40)
    
    # Display history
    print("📚 STUDENT PREVIOUS COURSE HISTORY:")
    print("-" * 50)
    for course_id, grade in sorted(student['history'].items()):
        score = GRADE_MAPPING[grade]
        print(f"  Course {course_id:3d} → Grade: {grade:2s} (Score: {score:.1f})")
    
    # Derive student latent vector
    student_vector = derive_student_latent_vector(student, course_factors, latent_cols)
    
    print(f"\n🧠 DERIVED STUDENT LATENT VECTOR:")
    print("-" * 50)
    for i, factor in enumerate(latent_cols):
        print(f"  {factor:35s}: {student_vector[i]:.4f}")
    
    # Display offered electives
    print(f"\n📋 OFFERED ELECTIVES (40 courses from 120):")
    print("-" * 50)
    oes = student['offered_electives']
    for i in range(0, len(oes), 8):
        print(f"  {', '.join(str(c) for c in oes[i:i+8])}")
    
    return {
        'student_num': student_num,
        'history': student['history'],
        'offered_electives': student['offered_electives'],
        'latent_vector': student_vector,
        'latent_factor_names': latent_cols
    }

# Example: Create 3 student queries
if __name__ == "__main__":
    student_queries = []
    for i in range(1, 4):
        query = student_query(student_num=i)
        student_queries.append(query)
    
    print(f"\n\n{'='*70}")
    print("SUMMARY: 3 Student Queries Created Successfully")
    print(f"{'='*70}\n")
