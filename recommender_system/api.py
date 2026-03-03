from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from data import grades_df, prof_feedback_df
from models import (
    build_course_embeddings,
    build_prof_embeddings,
    score_courses,
    score_professors,
    merge_scores,
)

app = FastAPI()

# precompute embeddings
student_course_emb, course_emb = build_course_embeddings(grades_df)
student_prof_emb, prof_emb = build_prof_embeddings(prof_feedback_df)


class ScoresResponse(BaseModel):
    scores: Dict[str, float]


class MergeRequest(BaseModel):
    course_scores: Dict[str, float]
    prof_scores: Dict[str, float]
    alpha: float = 0.7
    beta: float = 0.3


class MergeResponse(BaseModel):
    ranked: List[List]  # list of [course, professor, score]


@app.get("/student/{student_id}")
def get_student_profile(student_id: int):
    if student_id not in grades_df.index:
        return {"error": "student not found"}
    # just return the row as dict
    return grades_df.loc[student_id].dropna().to_dict()


@app.get("/courses/{student_id}", response_model=ScoresResponse)
def get_course_scores(student_id: int, method: str = "svd"):
    # note: method is ignored here because embeddings are precomputed; in a
    # real service you'd rebuild or cache multiple versions.
    scores = score_courses(student_id, grades_df, student_course_emb, course_emb)
    return {"scores": scores}


@app.get("/professors/{student_id}", response_model=ScoresResponse)
def get_professor_scores(student_id: int):
    scores = score_professors(student_id, prof_feedback_df, student_prof_emb, prof_emb)
    return {"scores": scores}


@app.post("/merge", response_model=MergeResponse)
def merge_endpoint(req: MergeRequest):
    # optionally filter by teaching assignment if available
    try:
        from data import prof_course_map
    except ImportError:
        prof_course_map = None
    ranked = merge_scores(req.course_scores, req.prof_scores, req.alpha, req.beta,
                          course_prof_map=prof_course_map)
    # convert numpy types to native
    ranked_native = [[c, p, float(s)] for c, p, s in ranked]
    return {"ranked": ranked_native}
