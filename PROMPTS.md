What improvements can be made in terms of features? Example of new routes?
Put yourself in the role of a Data Scientist / MLOps engineer. What features would be interesting and easy to implement without making things unnecessarily complex? Keep in mind the initial objective of the project — do not propose features that are unrelated or useless for 80% of users. If all necessary features are already present, do not invent new ones. Rank your ideas by priority and implementation difficulty. Write the result in a new ROADMAP_VX.md file — all features from previous roadmaps have already been implemented. Propose API features and/or Streamlit app features that are useful for users. Propose only useful and necessary things! If the project is already complete and your proposals only add complexity without value, it is better to avoid.
If you still have proposals, explain in detail the why and how.

------------------------------------


In plan mode:
From the git history, update the documentation in .md files including readme.md to integrate new features. Also explain how to clone the project and run docker compose. A doc to explain how to use the tool for a beginner with Python code examples, etc.
If some .md documentation files are already complete and perfect, no need to modify them. Add Anthropic_api_key in docker compose.

------------------------------------

------------------------------------
Think in plan mode:
From the latest PRs (one week) and the documentation:
1. Add the missing unit tests.
2. Add the integration tests needed to validate the overall product behavior.
Add end-to-end (e2e) tests to validate the overall product behavior.
Determine the test coverage rate before and after your plan.



Is the MLflow integration 100% working? For example, are all retrains stored in MLflow experiments with the correct KPIs, etc.? I want MLflow features to be used 100%. Create an implementation plan to make that the case.




Create a Streamlit help page with an LLM chat to help the user from a code perspective: generating train scripts with sklearn mlflow..., API calls to the solution, help using the Streamlit app, explanation of the various indicators, etc. Base responses on the project documentation and skills. Add documentation if needed. Possibility to view .md docs directly.



As a security expert, analyze the codebase to identify security issues. Verify that no real environment variables are published in git.
Create a plan to improve project security.



Create the missing useful tests for the following scripts:
- services/golden_test_service.py
- api/models.py
- api/monitoring.py
- api/predictions.py
- api/users.py
- db/database.py
- src/main.py

Give the overall coverage rate after the additions.




Manual action required
GitHub Actions: create these 5 secrets in the repository settings (Settings → Secrets → Actions):
CI_DB_PASSWORD
CI_MINIO_ACCESS_KEY
CI_MINIO_SECRET_KEY
CI_SECRET_KEY
CI_ADMIN_TOKEN
