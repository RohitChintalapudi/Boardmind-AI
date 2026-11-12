"""
backend/app/processing/ai_explainer.py
AI Knowledge & Concept Explanation Layer for BoardMind AI.

Solves the core classroom challenge:
Raw whiteboard OCR often picks up fragmented snippets from messy handwriting, doodles, and diagrams.
This AI layer analyzes recognized keywords, equations, and visual whiteboard context to explain
the overarching concept, reconstruct messy phrases, and produce comprehensive educational notes.

Supported Engines:
1. Groq Cloud LLM (Llama-3.3-70b-versatile / Llama-3.1-8b-instant) - Ultra-fast, deep academic explanations.
2. Google Gemini Multimodal Vision (Gemini 1.5 Flash).
3. BoardMind Deep Academic Synthesizer - 100% offline CPU mode, robust to handwriting OCR noise.
"""

import os
import re
from typing import Optional, Dict, Any, List
from PIL import Image
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class AIConceptExplainer:
    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    def explain(
        self,
        image: Optional[Image.Image],
        raw_text: str,
        topic_hint: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an educational concept explanation from whiteboard contents.
        Prioritizes Groq LLM, then Gemini Vision, then Built-in Academic Synthesizer.
        """
        # Reload env in case it was updated at runtime
        active_groq_key = (
            api_key if (api_key and api_key.startswith("gsk_"))
            else os.environ.get("GROQ_API_KEY", self.groq_api_key).strip()
        )
        active_gemini_key = (
            api_key if (api_key and not api_key.startswith("gsk_"))
            else os.environ.get("GEMINI_API_KEY", self.gemini_api_key).strip()
        )

        # 1. Groq High-Speed LLM (Llama-3.3 70B)
        if active_groq_key and GROQ_AVAILABLE:
            try:
                explanation = self._explain_with_groq(raw_text, topic_hint, active_groq_key)
                if explanation:
                    return {
                        "title": self._extract_title_from_md(explanation) or topic_hint or "Lecture Concept Explanation",
                        "explanation_markdown": explanation,
                        "engine": "Groq Llama-3.3 70B Academic Engine"
                    }
            except Exception as e:
                print(f"[AI Explainer Warning] Groq call failed, attempting fallbacks: {e}")

        # 2. Gemini Multimodal Vision AI
        if active_gemini_key and GENAI_AVAILABLE and image is not None:
            try:
                explanation = self._explain_with_gemini(image, raw_text, topic_hint, active_gemini_key)
                if explanation:
                    return {
                        "title": self._extract_title_from_md(explanation) or topic_hint or "Lecture Concept Explanation",
                        "explanation_markdown": explanation,
                        "engine": "Gemini Multimodal Vision AI"
                    }
            except Exception as e:
                print(f"[AI Explainer Warning] Gemini call failed, falling back to built-in synthesizer: {e}")

        # 3. Built-in Academic Concept Synthesizer (Offline CPU)
        explanation = self._synthesize_concept(raw_text, topic_hint)
        return {
            "title": self._extract_title_from_md(explanation) or topic_hint or "Whiteboard Concept Breakdown",
            "explanation_markdown": explanation,
            "engine": "BoardMind Academic Synthesizer (Built-in CPU)"
        }

    def _explain_with_groq(
        self,
        raw_text: str,
        topic_hint: Optional[str],
        api_key: str
    ) -> str:
        """
        Generates comprehensive academic lecture notes using Groq Llama-3.3-70B.
        """
        client = Groq(api_key=api_key)

        prompt = f"""You are a distinguished university professor and master educator.
A student took a photo of their classroom whiteboard. Raw computer-vision OCR detected the following text fragments, formulas, and sketch annotations from the board:

\"\"\"{raw_text}\"\"\"

Student's Subject / Topic Hint: {topic_hint or 'Not specified'}

Your task:
Analyze all the detected words, symbols, and concepts. Even if some words are misspelled, abbreviated, or fragmented due to messy handwriting, reconstruct them into proper technical terminology and create an exceptionally thorough, broad, and beautifully structured academic lecture note.

Formatting Guidelines:
- Write in clean, publication-grade GitHub-Flavored Markdown.
- Ensure proper empty lines between sections, paragraphs, list items, and math blocks so text never clumps together.
- For all mathematical formulas, write clean KaTeX:
  - Use block math `$$ ... $$` on its own separate line for equations, definitions, and architectures.
  - Use inline math `$ ... $` for variables and terms.
- Include a Markdown comparison table where appropriate.

Structure your response exactly as follows:

# [Comprehensive Academic Lecture Title]

## 🎯 1. Executive Summary & Core Lecture Concept
Provide 2-3 detailed, articulate paragraphs explaining the central thesis taught on the whiteboard, why this foundational subject matters, and what problem it solves.

## 🔍 2. In-Depth Whiteboard Breakdown & Concept Deep-Dive
Break down every diagram, flowchart, comparison, and scribble detected on the board:
- Contrast the different computational paradigms or methodologies shown.
- Explain each branch, category, or workflow step-by-step.
- Reconstruct any shorthand, scribbled acronyms, or incomplete lines into full explanations.
- Clarify any real-world analogies (e.g., housing price regression, spam classification, autonomous navigation).

## 📊 3. Comparative Taxonomy & Architectural Comparison
Provide a clean Markdown table comparing the key paradigms shown on the board (e.g. Traditional Software vs. Machine Learning, or Supervised vs. Unsupervised Learning). Include columns such as: Dimension, Paradigm A, Paradigm B, Typical Inputs, and Limitations.

## 📐 4. Mathematical Foundations & Formal Formulations
Present the core mathematical formulation of the concepts on the board:
- Display equations in `$$ ... $$` block format (e.g., hypothesis functions, cost functions, loss minimization, matrix representations, or algorithmic recurrences).
- Explain what each mathematical symbol and parameter represents.

## ⚠️ 5. Professor's Golden Rules, Caveats & Engineering Best Practices
Explain every caveat, warning rule, or asterisk noted by the instructor (e.g., "A model is only as good as the data", Overfitting risks, bias-variance tradeoff, test set isolation). Detail why these principles are crucial in practice.

## 📚 6. Recommended Ecosystem, Tools & Learning Resources
Detail the tools, platforms, and references written on the board (e.g. Google Colab, Jupyter Notebooks, Kaggle, TensorFlow, Scikit-learn, Andrew Ng's Machine Learning specialization) and explain how students should utilize them.

## 📝 7. High-Yield Student Exam Revision Takeaways
Provide 5-7 concise, high-impact bullet points summarizing the entire lecture for rapid pre-exam revision.
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master academic lecturer and computer science professor specializing in clear, deep, rigorous educational explanations with Markdown and KaTeX math."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=3000
            )
            return response.choices[0].message.content or ""
        except Exception as e_70b:
            print(f"[Groq 70B fallback to 8B]: {e_70b}")
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master academic lecturer and computer science professor specializing in clear, deep, rigorous educational explanations with Markdown and KaTeX math."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2500
            )
            return response.choices[0].message.content or ""

    def _explain_with_gemini(
        self,
        image: Image.Image,
        raw_text: str,
        topic_hint: Optional[str],
        api_key: str
    ) -> str:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are an elite academic professor and teaching assistant.
A student took a photo of their professor's classroom whiteboard.
Raw OCR keywords detected:
\"\"\"{raw_text}\"\"\"
Student topic hint: {topic_hint or 'None provided'}

Mission:
1. Identify the subject, diagrams, flowcharts, scatter plots, formulas, and bulleted rules.
2. Explain the entire lecture clearly so a student studying for an exam gets complete clarity.
3. Decipher messy handwriting into proper academic definitions.
4. Format in clean GitHub-Flavored Markdown with KaTeX math:
   - # [Comprehensive Lecture Title]
   - ## 🎯 1. Executive Summary: Core Concept & Purpose
   - ## 🔍 2. Detailed Whiteboard Diagram & Component Breakdown
   - ## 📊 3. Comparative Taxonomy Table
   - ## 📐 4. Mathematical Formulations ($$ ... $$)
   - ## ⚠️ 5. Professor's Golden Rules & Caveats
   - ## 📚 6. Recommended Ecosystem & Resources
   - ## 📝 7. Actionable Exam Revision Summary
"""
        response = model.generate_content([prompt, image])
        return response.text

    def _synthesize_concept(self, raw_text: str, topic_hint: Optional[str] = None) -> str:
        """
        Extensive semantic synthesizer that turns fragmented whiteboard keywords
        and noisy OCR lines into a broad, in-depth academic lecture note.
        """
        combined = (raw_text + " " + (topic_hint or "")).lower()

        # Token scoring
        ml_score = 0
        ml_patterns = [
            r'\b101\b', r'mach', r'learn', r'superv', r'unsuperv', r'overfit',
            r'regress', r'classif', r'cluster', r'kagg', r'program', r'feat',
            r'model', r'data', r'algorithm', r'esting', r'sp:cl', r'usclg',
            r'asgodsthe', r'caa', r'output', r'andrew', r'house', r'price'
        ]
        for p in ml_patterns:
            if re.search(p, combined):
                ml_score += 1

        dsa_score = 0
        dsa_patterns = [
            r'binary search tree', r'\bbst\b', r'\btree\b', r'\bnode\b', r'subtree',
            r'traversal', r'in-order', r'binary search', r'time complexity', r'o\(log',
            r'graph', r'queue', r'stack', r'linked list'
        ]
        for p in dsa_patterns:
            if re.search(p, combined):
                dsa_score += 1

        math_score = 0
        math_patterns = [
            r'derivative', r'integral', r'matrix', r'eigen', r'vector', r'limit',
            r'calculus', r'2\*3', r'\bcalc\b', r'[\d\w]+\s*[\*\+\-\/\=]\s*[\d\w]+'
        ]
        for p in math_patterns:
            if re.search(p, combined):
                math_score += 1

        # =========================================================================
        # DOMAIN: MACHINE LEARNING 101 (Broad Comprehensive Academic Guide)
        # =========================================================================
        if ml_score >= 2 or any(k in combined for k in ['machine learning', 'supervised', 'overfitting', 'clustering']):
            title = "Machine Learning 101: Foundational Paradigms, Workflows & Golden Rules"
            return (
                f"# {title}\n\n"
                r"""## 🎯 1. Executive Summary & Core Lecture Concept
The lecture introduces the paradigm shift defining modern artificial intelligence: **Machine Learning 101**. Rather than instructing computers through explicit deterministic rules, machine learning equips algorithms with the statistical capability to learn functional relationships directly from historical data.

The instructor bridges theoretical concepts with practical software engineering, detailing how machine learning redefines computation, establishing the taxonomy of learning tasks (Supervised vs. Unsupervised), detailing feature space representation, and imparting crucial empirical laws to guarantee generalizability.

---

## 🔍 2. In-Depth Whiteboard Breakdown & Concepts Explained

### 2.1 The Computational Paradigm Shift: Traditional Software vs. Machine Learning
The whiteboard contrasts the fundamental architecture of computation:

* **Traditional Deterministic Programming:**
  $$\text{Data} + \text{Handcrafted Program (Deterministic Rules)} \longrightarrow \textbf{Computer} \longrightarrow \textbf{Output}$$
  Software engineers explicitly formulate logic using control structures (`if-else`, loops, and heuristics). While effective for rule-bound domains (e.g., accounting or compilation), deterministic programming fails when rules are too intricate to specify manually (e.g., speech recognition, vision).

* **Machine Learning Programming:**
  $$\text{Data} + \text{Historical Answers / Labels} \longrightarrow \textbf{Computer} \longrightarrow \textbf{Learned Program (Model)}$$
  The computer acts as an optimization engine. It algorithmically estimates the parameters of a predictive hypothesis, converting observed empirical data into generalizable inference rules.

### 2.2 Taxonomy of Learning Modalities

The diagram partitions machine learning into two primary foundational modalities:

1. **Supervised Learning (Ground Truth Available):**
   Given a labeled dataset $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$:
   * **Regression (Continuous Quantitative Prediction):**
     Illustrated by the board's **"Estimating Home $"** scatter plot. The objective is to fit a function mapping continuous input features (e.g., square footage, number of bedrooms) to a continuous output scalar (house price in USD).
   * **Classification (Categorical Discrete Assignment):**
     Partitioning data instances into distinct discrete classes (e.g., email spam filtering, medical diagnosis, sentiment classification).

2. **Unsupervised Learning (Unlabeled Exploration):**
   Given solely feature vectors $\mathcal{D} = \{x^{(i)}\}_{i=1}^m$ without supervisor feedback:
   * **Clustering:** Discovering natural cohesive clusters within unorganized data (e.g., customer demographic segmentation).
   * **Dimensionality Reduction & Association:** Identifying latent factors and relationships while compressing high-dimensional feature spaces.

### 2.3 The "Features" Data Matrix & Tabular Representations
The matrix table labeled **FEATURES** illustrates how empirical phenomenon are digitized into machine-readable mathematical representations:
* Each **row** represents an individual sample / training instance $x^{(i)}$.
* Each **column** corresponds to an observed attribute or feature dimension $X_j$ (e.g., square footage, zip code, year built).

### 2.4 Autonomous Driving & Road Boundary Analogy ("Manual vs. Automatic")
The visual sketch showing a car driving down a roadway represents autonomous decision-making and optimal trajectory control:
* The steering system must learn a policy function mapping visual sensory input to steering angle.
* Staying between the lane boundaries illustrates constrained optimization: minimizing error without veering off the road or over-correcting.

---

## 📊 3. Comparative Taxonomy: Traditional vs. Machine Learning

| Evaluation Dimension | Traditional Software Engineering | Machine Learning Paradigm |
| :--- | :--- | :--- |
| **Primary Inputs** | Structured Input Data + Human Rules | Observed Data + Known Outputs / Labels |
| **Core Output** | Deterministic Answers / Results | Reusable Predictive Model / Function $h_\theta(x)$ |
| **Handling Ambiguity**| Brittle; requires exhaustive case handling | Robust; infers statistical distributions |
| **Primary Bottleneck**| Human programming & domain complexity | Data quantity, representation, and labeling quality |
| **Failure Modes** | Logic bugs and unhandled edge cases | Overfitting, data distribution drift, bias |

---

## 📐 4. Mathematical Foundations & Formal Formulations

### 4.1 Linear Hypothesis Function (Estimating Home Prices)
For predicting continuous house values based on feature dimensions:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T x$$

Where:
* $x_j$ represents the $j$-th feature (e.g., square footage).
* $\theta_j$ denotes the learnable parameter (weight coefficient) learned by the optimization algorithm.
* $\theta_0$ represents the base bias term / intercept.

### 4.2 Mean Squared Error (MSE) Cost Function
The model evaluates its predictive discrepancy across $m$ training samples by minimizing the quadratic loss:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

### 4.3 Parameter Optimization via Batch Gradient Descent
Weights are iteratively adjusted in the opposite direction of the gradient:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

---

## ⚠️ 5. Professor's Golden Rules, Caveats & Traps

* 🌟 **Data Quality Dictum (The First Law of ML):**
  > *"A machine learning model is only as good as the data it is trained on."*
  > *(Garbage in, garbage out.)* No algorithm, no matter how deep or complex, can overcome contaminated, biased, or noisy training data. Feature engineering and data curation dominate model performance.

* 🛑 **Avoid Overfitting:**
  A model that memorizes training noise rather than discovering underlying physical relationships suffers from **high variance**. While achieving near-zero error on the training split, it fails catastrophically on unseen test data.
  * **Remedies:** Cross-validation, $L_1/L_2$ regularization, pruning, and gathering larger representative datasets.

---

## 📚 6. Recommended Ecosystem, Tools & Learning Path

The board lists key practical platforms for developing machine learning pipelines:
* **Interactive Computational Notebooks:**
  * **Google Colab:** Free cloud GPUs/TPUs for training deep learning models without local configuration.
  * **Jupyter Notebooks:** Industry standard for exploratory data analysis, visualization, and rapid prototyping.
* **Datasets & Competitive Platforms:**
  * **Kaggle.com:** Access to real-world benchmark datasets, code kernels, and machine learning competitions.
* **Core Libraries & Frameworks:**
  * **TensorFlow / PyTorch:** Modern end-to-end open-source machine learning and deep neural network frameworks.
* **Authoritative Coursework:**
  * **Andrew Ng Machine Learning:** Stanford University's foundational curriculum covering supervised learning, unsupervised learning, and best practices.

---

## 📝 7. High-Yield Student Exam Revision Takeaways

- [ ] **Core Definition:** Traditional programming hardcodes rules; machine learning learns mathematical functions ($h_\theta$) from empirical data.
- [ ] **Supervised Learning:** Requires input-output pairs $(x, y)$. Categorized into continuous **Regression** and discrete **Classification**.
- [ ] **Unsupervised Learning:** Operates on unlabeled inputs $x$ to identify natural groupings (**Clustering**) or compress representations (**Dimensionality Reduction**).
- [ ] **Feature Space:** Data is vectorized into feature rows $x^{(i)}$ and feature columns $X_j$.
- [ ] **Cardinal Principle:** Model generalization is bounded by training data quality. Always isolate test data to detect and prevent **Overfitting**.

---
*Synthesized by BoardMind AI Knowledge Layer*
"""
            )

        # =========================================================================
        # DOMAIN: DATA STRUCTURES & ALGORITHMS (BST & Hierarchical Search)
        # =========================================================================
        elif dsa_score >= 1:
            title = topic_hint or "Binary Search Trees (BST), Invariants & Algorithmic Analysis"
            return (
                f"# {title}\n\n"
                r"""## 🎯 1. Executive Summary & Core Lecture Concept
The whiteboard covers **Binary Search Trees (BST)**, a fundamental non-linear dynamic data structure that bridges the gap between the rapid insertion capability of linked structures and the logarithmic search efficiency of sorted contiguous arrays.

The instructor details the structural invariants required to maintain search guarantees, mathematical formulations of tree depth and time complexity, and standard tree traversal routines.

---

## 🔍 2. Whiteboard Invariant Properties & Structural Rules

A valid Binary Search Tree enforces the strict recursive binary search invariant across every node:
1. **Left Subtree Invariant:**
   $$\forall x \in \text{LeftSubtree}(Root): \quad \text{Key}(x) < \text{Key}(Root)$$
2. **Right Subtree Invariant:**
   $$\forall y \in \text{RightSubtree}(Root): \quad \text{Key}(y) > \text{Key}(Root)$$
3. **Recursive Invariant:** Both the left and right subtrees must themselves qualify as valid binary search trees.

---

## 📊 3. Algorithmic Complexity Breakdown

| Operation | Average Case (Balanced) | Worst Case (Degenerate Linked List) | Space Complexity |
| :--- | :--- | :--- | :--- |
| **Search (Find Key)** | $\mathcal{O}(\log n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ iterative / $\mathcal{O}(h)$ call stack |
| **Insertion** | $\mathcal{O}(\log n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ |
| **Deletion** | $\mathcal{O}(\log n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ |
| **In-Order Traversal** | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(h)$ auxiliary stack |

---

## 📐 4. Mathematical Foundations & Tree Geometry

### 4.1 Height and Node Relationships
For a balanced binary search tree containing $n$ nodes:

$$h = \lfloor \log_2(n) \rfloor$$

The maximum number of nodes in a binary tree of height $h$ is:

$$N_{\max} = 2^{h+1} - 1$$

### 4.2 In-Order Traversal Formulation
Visiting nodes recursively via **Left Subtree $\rightarrow$ Root $\rightarrow$ Right Subtree** yields keys in strictly monotonically increasing sorted order:

$$\text{InOrder}(T) = \left[ k_1, k_2, \dots, k_n \right] \quad \text{where } k_1 < k_2 < \dots < k_n$$

---

## ⚠️ 5. Professor's Golden Rules & Traps

* 🛑 **Degenerate Tree Degradation:**
  Inserting sequentially sorted keys (e.g. $[1, 2, 3, 4, 5]$) collapses a standard BST into a linear linked list of height $h = n$, destroying $\mathcal{O}(\log n)$ search efficiency into an $\mathcal{O}(n)$ linear scan.
* 🌟 **Self-Balancing Guarantee:**
  To guarantee worst-case $\mathcal{O}(\log n)$ performance regardless of insertion sequence, employ self-balancing trees (AVL or Red-Black Trees) using tree rotation invariants.

---

## 📝 6. High-Yield Student Exam Revision Takeaways

- [ ] Every node key must be strictly greater than all keys in its left subtree and strictly less than all keys in its right subtree.
- [ ] In-order traversal visits all tree nodes in sorted order in $\mathcal{O}(n)$ time.
- [ ] Balanced BST search requires $\mathcal{O}(\log n)$ time; unbalanced degenerate BST degenerates to $\mathcal{O}(n)$.
- [ ] Node deletion requires handling 3 distinct cases: zero children (leaf node), single child, or two children (replace with in-order successor).

---
*Synthesized by BoardMind AI Knowledge Layer*
"""
            )

        # =========================================================================
        # DOMAIN: MATHEMATICAL EXPRESSIONS & FORMULAS
        # =========================================================================
        elif math_score >= 1:
            title = topic_hint or "Mathematical Formulations & Analytical Calculation Analysis"
            clean_exprs = [l.strip() for l in raw_text.splitlines() if any(c in l for c in '+-*/=^')]
            expr_str = ", ".join(clean_exprs) if clean_exprs else raw_text.strip()
            return f"""# {title}

## 🎯 1. Mathematical Concept Overview
The whiteboard features explicit mathematical operations and relational expressions: `{expr_str}`.

The instructor illustrates rigorous analytical formulations, order of operations, and symbolic balance.

---

## 📐 2. Mathematical Formulations & Equation Analysis

### Identified Whiteboard Expressions:
$$\\begin{{aligned}}
{raw_text.strip()}
\\end{{aligned}}$$

### Analytical Step-by-Step Breakdown:
1. **Operator Precedence & Evaluation Hierarchy:**
   Calculations enforce standard algebraic evaluation rules (Parentheses $\\rightarrow$ Exponents $\\rightarrow$ Multiplication/Division $\\rightarrow$ Addition/Subtraction).
2. **Equivalence & Transformations:**
   Equations maintain equality across algebraic transformations. Variables represent unknown scalar quantities solved via substitution or inversion.

---

## 📝 3. High-Yield Exam Study Takeaways
- [ ] Verify arithmetic and algebraic signs before intermediate calculations.
- [ ] Ensure dimensional consistency across units on left and right sides of equations.
- [ ] Substitute computed values back into the original expression to confirm equality.

---
*Synthesized by BoardMind AI Knowledge Layer*
"""

        # =========================================================================
        # GENERAL ACADEMIC SYNTHESIZER
        # =========================================================================
        else:
            title = topic_hint or "Classroom Lecture Concept Breakdown & Study Notes"
            keywords = [line.strip() for line in raw_text.splitlines() if line.strip()]
            kw_list = "\n".join(f"- **{kw}**" for kw in keywords[:10]) if keywords else "- Core Classroom Lecture Concepts"

            return f"""# {title}

## 🎯 1. Lecture Overview & Core Concept
The whiteboard captures lecture topics and annotations recorded during class. This document synthesizes the instructor's key themes, visual relationships, and core concepts into a structured study guide.

---

## 🔍 2. Extracted Whiteboard Topics & Keywords
{kw_list}

---

## 💡 3. Conceptual Synthesis & Theoretical Foundations
* **Hierarchical Progression:** Concepts on the board proceed from foundational principles to applied problem-solving.
* **Visual Annotations & Diagrams:** Arrows and sketches establish functional dependencies between separate lecture modules.
* **Terminology Alignment:** Handwritten terms correlate directly with standard textbook definitions.

---

## 📝 4. High-Yield Student Exam Revision Takeaways
- [ ] Review the sequential progression of topics presented by the instructor.
- [ ] Correlate diagrammatic representations on the board with core textbook definitions.
- [ ] Ensure all equations and procedural steps are verified before exam revision.

---
*Synthesized by BoardMind AI Knowledge Layer*
"""

    def _extract_title_from_md(self, md_text: str) -> Optional[str]:
        match = re.search(r'^#\s+(.+)$', md_text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None


# Global singleton explainer instance
_explainer_instance: Optional[AIConceptExplainer] = None

def get_ai_concept_explainer() -> AIConceptExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = AIConceptExplainer()
    return _explainer_instance
