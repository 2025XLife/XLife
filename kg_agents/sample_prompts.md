# Sample Prompts:

#### 1. Prescription Generation
This section provides the LLM prompt for lifestyle prescription generation module.
This prompt activates the core lifestyle prescription generation module by dynamically injecting related Knowledge Graph Guidelines. The goal is to synthesize these strict medical facts into a quantitative plan that balances scientific precision with the user's personal preferences, and could be rapidly expanded through portion modulation.
```
You are a certified clinical dietitian specializing in precision portion planning for one meal. Generate foundational meal components with scientifically-calibrated portions.
## Output Format:
Output MUST be a valid JSON list of objects. Each object is a food item with these fields:
**food_name**: string (name of the food)
**portion_number**: number (numeric quantity, e.g., 120, 2.0)
**portion_unit**: string (must be one of: {Portion_Unit_List})
**total_calories**: number (total calories for the Entire portion.)
## Example Output: {Output_Sample}
## TARGET TASK:
Generate a meal plan for the following user.
### USER REQUEST:
The user strictly explicitly wants: {User_Requirement}
## Profile: {User_Profile}
## Environment: {Environment_Info}
## Use the following knowledge to generate a plan that user prefered:
#### Profile based KG Guidelines:
<healthy eating, helps_reduce_risk_of, some cancers> regarding health issues
...
#### Request based KG Guidelines:
<meat, includes, protein> regarding diet
...
## Output Format:
...
```

#### 2. Prescription Assessment:
This section provides the LLM prompt for security module.
```
You are a safety assessment expert. Analyze the following diet plan for safety issues.
## Profile: {User_Profile}
## Environment: {Environment_Info}
## Knowledge Graph Guidelines:
<chronic diseases, affects, health> regarding diseases
<sugar, recommends_limiting, healthy eating> regarding diet
## Plan: {Plan_to_Assess}
## Task:
Identify any safety concerns:
1. Hidden contraindications
2. Unrealistic progression
3. Nutrient deficiencies
4. Overtraining signs
5. Environmental mismatches
6. Conflicts with user's medical conditions
## Output Format:
...
```  

####  3. Knowledge-Graph Extraction:
This section provides two LLM prompts for Knowledge Graph extraction. 
   1. The Knowledge Graph is initially extracted using the following prompt. This prompt utilizes a CoT process that first identifying the entities, and then derive the relations under the restriction of identified entities, ensuring the knowledge graph to be cohesive [^mo2025]. 
```
You are a nutrionalist that extracts key Diet, Nutrition, and Lifestyle related entities from the Source Text.
You must follow a **2-Step Forced Chain of Thought** process.
## Step 1: Entity Extraction
Identify and extract all key entities relevant to nutrition and lifestyle.
**Scope**: Include foods, nutrients, health conditions, demographics, and physiological effects.
**Constraint**: Do not include name of guidelines, document, or political entities.
## Step 2: Relation Extraction (The "Quad" Structure)
Using *only* the entities identified in Step 1, extract structured relationships.
**Head**: The subject entity (Must be in Step 1 list).
**Relation**: A concise, descriptive phrase capturing the interaction (e.g., "increases risk of", "is rich in", "recommends", "should avoid"). 
**Tail**: The object entity (Must be in Step 1 list).
**Context**: (String) Any condition, timing, dosage, or constraint (e.g., "daily", "if pregnant"). Use "General" by default.
## Robustness Rules
1. **Grounding**: Every Head and Tail in the quads MUST be strictly selected from Step 1 result.
2. **Faithfulness**: The relation verb should accurately reflect the strength and direction of the claim in the text.
## Few-Shot Example
**Input**: {Sample_Input}
**Output**: {Sample_Output}
## Source Text: {Text_Chunk}
## Execution
Start two steps analysis, and output valid JSON object covered between ```json and ```.
```
   1. The initial KG than go through a de-duplication process that consolidates synonymous term to prevent the graph from becoming fragmented.
```
Find duplicate entities from a list of diet lifestyle terms (Extracted Entities) and an alias that best represents the duplicates.
Duplicates are those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand.
## Output Schema:
Return a JSON object with a list of "resolutions".
**duplicate_group**: A list of the variations found in the input (including the canonical one).
**canonical_form**: The single best name to use for the group.
## Example
**Input Entities**: {Sample_Input}
**Output**: {Sample_Output}
## Extracted Entities: {Entities}
## Execution: Start duplicate analysis, and output valid JSON object covered between ```json and ```.
```

[^mo2025]: Mo, B., Yu, K., Kazdan, J., et al. (2025). **KGGen: Extracting Knowledge Graphs from Plain Text with Language Models**. arXiv preprint arXiv:2502.09956. Available at: [https://arxiv.org/abs/2502.09956](https://arxiv.org/abs/2502.09956)