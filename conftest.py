import os
import hypothesis

# This setup is drawn from Astropy and might not be entirely relevant to us;
# in particular we don't have a cron run for slow tests.

# Tell Hypothesis that we might be running slow tests, to print the seed blob
# so we can easily reproduce failures from CI, and derive a fuzzing profile
# to try many more inputs when we detect a scheduled build or when specifically
# requested using the HYPOTHESIS_PROFILE=fuzz environment variable or
# `pytest --hypothesis-profile=fuzz ...` argument.

hypothesis.settings.register_profile("interactive", deadline=None)
hypothesis.settings.register_profile(
    "ci", deadline=None, print_blob=True, derandomize=True
)
hypothesis.settings.register_profile(
    "fuzzing", deadline=None, print_blob=True, max_examples=1000
)
default = (
    "fuzzing"
    if (
        os.environ.get("IS_CRON") == "true"
        and os.environ.get("ARCH_ON_CI") not in ("aarch64", "ppc64le")
    )
    else "interactive"
)  # noqa: E501
hypothesis.settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", default))
