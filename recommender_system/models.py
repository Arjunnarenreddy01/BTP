import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_course_embeddings(grades_df, n_components=2):
    """Factorize the student-course grade matrix using SVD.

    Returns student_embeddings (num_students x n_components) and
    course_embeddings (num_courses x n_components).
    """
    # fill missing values with student mean or 0
    mat = grades_df.fillna(0).values
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    student_emb = svd.fit_transform(mat)
    course_emb = svd.components_.T
    return student_emb, course_emb


def score_courses(student_id, grades_df, student_emb, course_emb):
    """Compute cosine similarity between one student and all course vectors."""
    import numpy as np

    idx = list(grades_df.index).index(student_id)
    s_vec = student_emb[idx]
    # cosine similarity
    norms = np.linalg.norm(course_emb, axis=1) * np.linalg.norm(s_vec)
    sims = course_emb.dot(s_vec) / norms
    # build dict course->score
    return dict(zip(grades_df.columns, sims))


def build_prof_embeddings(feedback_df, n_components=2):
    """Factorize student-professor rating matrix similarly."""
    mat = feedback_df.fillna(0).values
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    student_emb = svd.fit_transform(mat)
    prof_emb = svd.components_.T
    return student_emb, prof_emb


def score_professors(student_id, feedback_df, student_emb, prof_emb):
    idx = list(feedback_df.index).index(student_id)
    s_vec = student_emb[idx]
    norms = np.linalg.norm(prof_emb, axis=1) * np.linalg.norm(s_vec)
    sims = prof_emb.dot(s_vec) / norms
    return dict(zip(feedback_df.columns, sims))


def merge_scores(course_scores, prof_scores, alpha=0.7, beta=0.3):
    """Combine two score dictionaries into ranked list of (course,prof,score)."""
    merged = []
    for c, cs in course_scores.items():
        for p, ps in prof_scores.items():
            merged.append((c, p, alpha * cs + beta * ps))
    # sort descending
    return sorted(merged, key=lambda x: x[2], reverse=True)
