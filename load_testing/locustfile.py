"""Closed-loop beam-5 translation load with short English sentences."""

from itertools import cycle

from locust import HttpUser, constant, task

SENTENCES = (
    "The train is late.",
    "Please turn off the light before you leave.",
    "We will meet at the station tomorrow morning.",
    "The children are playing in the garden after school.",
    "Could you tell me how to get to the hospital?",
    "Farmers in the village are waiting for the rain to start.",
    "She bought vegetables and fruit at the market yesterday.",
    "The government announced a new scheme for students today.",
)


class TranslationUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.sentences = cycle(SENTENCES)

    @task
    def translate(self):
        with self.client.post(
            "/translate",
            json={"text": next(self.sentences), "beam_size": 5},
            headers={"User-Agent": "en-gu-load-test"},
            catch_response=True,
            timeout=50,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("Invalid JSON response")
                return
            if (
                not isinstance(body, dict)
                or not body.get("translation")
                or body.get("model_version") != response.headers.get("X-Model-Version")
            ):
                response.failure("Missing translation or inconsistent model version")
