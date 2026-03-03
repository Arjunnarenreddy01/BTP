import numpy as np
from sklearn.decomposition import TruncatedSVD

# optionally use PyTorch for neural collaborative filtering
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def _train_neural_embeddings(mat, n_components=2, epochs=500, lr=0.01):
    """Train simple matrix factorization using PyTorch.

    mat: numpy array with shape (num_students, num_courses)
    missing entries should be filled with 0 and a mask provided.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for neural CF but not installed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_students, num_courses = mat.shape

    # create mask of observed ratings (nonzero)
    mask = (mat != 0).astype(np.float32)
    ratings = torch.tensor(mat, dtype=torch.float32, device=device)
    mask = torch.tensor(mask, dtype=torch.float32, device=device)

    # embeddings as parameters
    student_emb = nn.Parameter(torch.randn(num_students, n_components, device=device) * 0.1)
    course_emb = nn.Parameter(torch.randn(num_courses, n_components, device=device) * 0.1)

    optimizer = torch.optim.Adam([student_emb, course_emb], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = student_emb @ course_emb.T
        loss = ((pred - ratings) * mask).pow(2).sum() / (mask.sum() + 1e-8)
        loss.backward()
        optimizer.step()
        # optional: print every 100 epochs
        if epoch % 100 == 0:
            print(f"neural CF epoch {epoch}, loss={loss.item():.4f}")

    return student_emb.detach().cpu().numpy(), course_emb.detach().cpu().numpy()


def build_course_embeddings(grades_df, n_components=2, method="svd"):
    """Return student and course embedding matrices.

    method can be "svd" or "neural". Neural uses a simple PyTorch MF model
    trained on observed grades.
    """
    # prepare matrix with zeros for missing grades
    mat = grades_df.fillna(0).values
    if method == "svd":
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        student_emb = svd.fit_transform(mat)
        course_emb = svd.components_.T
        return student_emb, course_emb
    elif method == "neural":
        return _train_neural_embeddings(mat, n_components=n_components)
    else:
        raise ValueError(f"unknown method {method}")


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


def build_prof_embeddings(feedback_df, n_components=2, method="svd"):
    """Return student and professor embedding matrices.

    method can be "svd" or "neural".
    """
    mat = feedback_df.fillna(0).values
    if method == "svd":
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        student_emb = svd.fit_transform(mat)
        prof_emb = svd.components_.T
        return student_emb, prof_emb
    elif method == "neural":
        return _train_neural_embeddings(mat, n_components=n_components)
    else:
        raise ValueError(f"unknown method {method}")


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
