from data import grades_df, prof_feedback_df
from models import (
    build_course_embeddings,
    build_prof_embeddings,
    score_courses,
    score_professors,
    merge_scores,
)


class RecommendationAgent:
    """Simple agent that reasons step-by-step and calls the model tools.

    You can choose the embedding `method` ("svd" or "neural").
    """

    def __init__(self, method="neural", n_components=2):
        self.method = method
        self.n_components = n_components
        # precompute embeddings once
        print(f"Agent: building course embeddings using {method}...")
        self.student_course_emb, self.course_emb = build_course_embeddings(
            grades_df, n_components=n_components, method=method
        )
        print(f"Agent: building professor embeddings using {method}...")
        self.student_prof_emb, self.prof_emb = build_prof_embeddings(
            prof_feedback_df, n_components=n_components, method=method
        )

    def recommend(self, student_id, top_k=5):
        print(f"Thought: need to score courses for student {student_id}")
        course_scores = score_courses(
            student_id, grades_df, self.student_course_emb, self.course_emb
        )
        print("Action: computed course scores")

        print(f"Thought: need to score professors for student {student_id}")
        prof_scores = score_professors(
            student_id, prof_feedback_df, self.student_prof_emb, self.prof_emb
        )
        print("Action: computed professor scores")

        print("Thought: merging scores into combined ranking")
        # if we know which professors teach which courses, only allow those
        from data import prof_course_map
        ranking = merge_scores(course_scores, prof_scores, course_prof_map=prof_course_map)
        print("Action: merged and sorted results")

        return ranking[:top_k]


if __name__ == "__main__":
    agent = RecommendationAgent()
    recs = agent.recommend(1, top_k=6)
    print("Final recommendation list (course, professor, score):")
    for tup in recs:
        print(tup)
