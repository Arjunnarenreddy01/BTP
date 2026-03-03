"""Simple HTTP-based agent that queries the FastAPI endpoints sequentially.

This imitates the behaviour of an LLM-powered agent by printing thought/action steps.
"""
import requests

# change the port to match the uvicorn server (default 8001 here)
BASE = "http://localhost:8001"


def get_student_profile(student_id):
    print(f"Thought: fetch profile for {student_id}")
    resp = requests.get(f"{BASE}/student/{student_id}")
    print("Action: got response")
    return resp.json()


def get_course_scores(student_id):
    print(f"Thought: compute course scores via API")
    resp = requests.get(f"{BASE}/courses/{student_id}")
    print("Action: received course scores")
    return resp.json()["scores"]


def get_professor_scores(student_id):
    print(f"Thought: compute professor scores via API")
    resp = requests.get(f"{BASE}/professors/{student_id}")
    print("Action: received professor scores")
    return resp.json()["scores"]


def merge(course_scores, prof_scores, alpha=0.7, beta=0.3):
    print("Thought: merge scores")
    resp = requests.post(f"{BASE}/merge", json={
        "course_scores": course_scores,
        "prof_scores": prof_scores,
        "alpha": alpha,
        "beta": beta,
    })
    print("Action: got merged ranking")
    return resp.json()["ranked"]


def run_agent(student_id):
    profile = get_student_profile(student_id)
    if "error" in profile:
        print(profile["error"])
        return

    cs = get_course_scores(student_id)
    ps = get_professor_scores(student_id)
    ranking = merge(cs, ps)
    print("Final recommendations:")
    for r in ranking[:5]:
        print(r)


if __name__ == "__main__":
    print("Start HTTP agent demo. Make sure uvicorn server is running.")
    run_agent(1)
