## Standard Workflow
1. First think through the problem, read the codebase for relevant files, and write a plan to projectplan.md.
2. The plan should have a list of todo items that you can check off as you complete them.
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please, every step of the way just give me a high level explanation of what changes you made.
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity, unless the simple solutions are not working.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project aims to automate the manual content analysis process from Quincy Lherisson's thesis using machine learning. The original research analyzed 462 news articles from 20 major US news sources to study how media frames leadership disparities at the intersection of gender and race. The goal is to build ML models that can automatically identify and classify these framing patterns in news articles.

## Original Research Context

The thesis examined news articles discussing leadership disparities, with human coders (9 research assistants working in pairs) manually identifying four types of frames:
- **Underrepresentation**: Groups appearing in leadership at LOWER rates than expected
- **Overrepresentation**: Groups appearing in leadership at HIGHER rates than expected  
- **Obstacles**: Systemic hindrances or barriers in pursuing leadership
- **Successes**: Achievements or accomplishments in pursuing leadership

These frames were analyzed across different demographic groups based on:
- Gender identity (men, women)
- Racial-ethnic identity (White people, people of color)
- Intersectional identities (e.g., White men, women of color)

## Key Documents

- `article_coding.md`: Contains the detailed coding guidelines and parameters used by human coders
- `quincy_thesis.md`: The full thesis document providing research methodology and findings

## Machine Learning Implementation Goals

When developing the ML system:

1. **Data Collection**: Build scrapers for the 20 news sources identified in the thesis (CNN, NPR, Yahoo News, etc.)
2. **Text Processing**: Develop NLP pipelines to identify relevant passages about leadership
3. **Frame Classification**: Train models to classify text segments into the four frame types
4. **Entity Recognition**: Identify which demographic groups are being discussed
5. **Counting System**: Replicate the "exemplar counting" approach from the manual coding

## Important Coding Principles from Original Research

- Code only **explicit** statements, not implied meanings
- Count each instance (exemplar) of framing individually
- Leadership context: business, government, and academia/education
- Maintain high inter-rater reliability (original study achieved average ICC of 0.75)

## Technical Considerations

- Consider using transformer-based models (BERT, RoBERTa) for frame classification
- Implement demographic entity recognition for intersectional analysis
- Build evaluation metrics that align with the original inter-rater reliability standards
- Create visualizations to replicate the statistical analyses from the thesis