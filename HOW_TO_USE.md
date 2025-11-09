# 🚀 ATLASky-AI User Guide

## How to Use the ATLASky-AI Demo

This guide shows you how to explore and test the ATLASky-AI verification system through the interactive dashboard.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Dashboard Overview](#dashboard-overview)
3. [Tab-by-Tab Guide](#tab-by-tab-guide)
4. [Testing Workflows](#testing-workflows)
5. [Understanding Results](#understanding-results)
6. [Advanced Features](#advanced-features)

---

## Quick Start

### Installation & Launch

```bash
# Navigate to project directory
cd AtlaSkI-AI

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Dashboard Overview

### Main Navigation Tabs

The dashboard has **6 main tabs**:

1. **📚 Methodology** - System architecture and theoretical foundations
2. **🗂️ STKG Structure** - Knowledge graph structure and domain examples
3. **💠 Verification Process** - Interactive three-stage verification pipeline
4. **🔄 AAIC Monitoring** - Adaptive parameter monitoring
5. **📊 Parameter Evolution** - Track how parameters change over time
6. **📜 Verification History** - Complete audit trail of all verifications

### Sidebar Controls

- **🤖 LLM Mode**: Choose between real OpenAI API or demo simulation
- **Testing Controls**: Domain and quality level selection for test fact generation
- **Batch Processing**: Generate and verify multiple facts automatically
- **Knowledge Graph Stats**: Current entities and relationships
- **AAIC Parameters**: Monitor adaptation settings
- **Performance Testing**: Introduce performance shifts for AAIC testing

---

## Tab-by-Tab Guide

### Tab 1: 📚 Methodology

**What You'll See:**
- Complete STKG formalization: G = (V, E, O, T, Ψ)
- Physics predicates: ψ_s (spatial), ψ_t (temporal), Ψ (combined)
- Error taxonomy: Content Hallucination, ST-Inconsistency, Semantic Drift
- Five-module verification pipeline (LOV, POV, MAV, WSV, ESV)
- AAIC adaptation mechanisms

**What It's For:**
Understanding the theoretical foundations and mathematical formulations behind ATLASky-AI.

---

### Tab 2: 🗂️ STKG Structure

**What You'll See:**
- **STKG Components**: V (entities), E (relationships), O (ontology), T (spatiotemporal), Ψ (physics)
- **Domain Examples**: See how STKGs differ across aerospace, healthcare, aviation, CAD
- **Live KG Status**: Current number of entities, relationships, and accepted facts
- **Recent Updates**: Last 5 accepted facts added to the knowledge graph
- **Ontology Browser**: Explore entity classes, relationships, physical constraints, domain rules

**What It's For:**
- Understanding how the knowledge graph is structured
- Seeing how verified facts integrate into the STKG
- Tracking knowledge graph growth as facts are verified

**Key Metrics Displayed:**
- Entities (V): Total entities in the graph
- Relationships (E): Total edges between entities
- Facts Verified: Total verifications run
- Accepted Facts: Facts that passed verification and were added to STKG

---

### Tab 3: 💠 Verification Process (Main Interactive Tab)

This is where you **test the complete verification pipeline**.

#### **Layout:**

**Left Column:** Pipeline stages (input)  
**Right Column:** Verification results (output)

#### **Workflow:**

##### **Step 0: Choose LLM Mode (Optional)**

**🤖 Use Real OpenAI API vs Demo Simulation**

Before generating test facts, decide whether to use real LLM extraction or demo mode:

**Option 1: Demo Simulation (Default - Recommended for Quick Testing)**
- ✅ No API key required
- ✅ Works offline
- ✅ Fast and reliable
- ✅ Generates realistic facts with honest confidence scoring
- ⚠️ Simulated extraction based on keyword matching and text quality

**Option 2: Real OpenAI API (For Production/Research)**
- ✅ Uses actual GPT-4o for fact extraction
- ✅ More sophisticated understanding
- ✅ Better handling of complex/ambiguous text
- ⚠️ Requires API key and costs money per call
- ⚠️ Requires internet connection

**How to Enable Real API:**
1. In sidebar, check **"🤖 Use Real OpenAI API"**
2. Enter your OpenAI API key (or set `OPENAI_API_KEY` environment variable)
3. System will use GPT-4o for Stage 2 (LLM Extraction)
4. Falls back to simulation if API call fails

**💡 Recommendation:** Start with demo mode to explore the system, then switch to real API for actual research/production use.

---

##### **Step 1: Generate Test Data (Sidebar) - IMPORTANT!**

⚠️ **You MUST generate a test fact BEFORE running the pipeline stages!**

**Why?** The "Generate Test Fact" button creates the initial raw text that flows through all three stages. Without this, you'll have no data to process.

**How to Generate:**

1. **Open the Sidebar** (left side of screen, click ">>" if collapsed)

2. **Find "Testing Controls" Section** (below LLM mode selection)

3. **Select Domain**:
   - **🏭 Aerospace**: Turbine blade measurements, tolerances, facility bays
   - **🏥 Healthcare**: Patient transfers, clinical protocols, care units
   - **✈️ Aviation**: Safety incidents, flight data, airspace
   - **🔧 CAD**: Assembly models, geometric constraints, parts

4. **Select Quality Level** (this determines what kind of raw text is generated):
   - **High Quality**: Perfect measurements, complete info, proper terminology
     - Example: "Installation completed in Bay 7. Blade Alpha measurement: 3.02 mm on leading edge. Nominal: 3.0 mm. Deviation: 0.01 mm. Tolerance: ±0.1 mm. Status: PASS."
     - Expected Result: ✅ **ACCEPT** (C ≈ 0.85)
   
   - **Medium Quality**: Minor spelling errors, good data
     - Example: "Instalation completed in bay 7. Blade Alpha measurment: 3.02 mm. Deviation: 0.02 mm."
     - Expected Result: ✅ **ACCEPT** (C ≈ 0.82)
   
   - **Spatial Issue**: Valid measurement but unclear/vague location
     - Example: "Blade inspection at somewhere in assembly area. Measured: 3.02 mm."
     - Expected Result: ⚠️ **REVIEW** or ❌ **REJECT**
   
   - **Semantic Issue**: Incorrect terminology, non-standard terms
     - Example: "Blade checkup. Size measurement was 3.02 millimeters. Expected around 3.0 mm."
     - Expected Result: ⚠️ **REVIEW** or ❌ **REJECT**
   
   - **Low Quality**: Vague, incomplete, minimal information
     - Example: "Blade part inspected. Measured approximately 3.5. Seems okay."
     - Expected Result: ❌ **REJECT** (C ≈ 0.0)

5. **Click "🎲 Generate Test Fact" Button**

**What Happens:**
- ✅ Raw text is generated based on your domain + quality selections
- ✅ Text includes quality-appropriate characteristics:
  - High quality: Complete data, proper terminology, specific values
  - Low quality: Vague descriptions, missing data, unclear references
- ✅ Text is automatically populated in Stage 1's "Raw Text Input" field
- ✅ Domain and location are automatically set
- ✅ Previous pipeline data is cleared (forces you to rerun stages)
- ✅ Blue indicator appears in Stage 1 showing "Test Fact Generated"

**Where to See the Generated Text:**
- Go to **Verification Process** tab
- Scroll to **Stage 1: Data Preprocessing**
- Expand "📊 Configure & Run Preprocessing"
- Look at the "📝 Raw Text Input" field
- You'll see the generated text matching your domain and quality selections!

**Example Visual Flow:**

```
Sidebar Selection:          Generated Raw Text:
├─ Domain: Aerospace   →   "Instalation completed in bay 7.
├─ Quality: Medium     →    Blade Gamma measurment: 3.02 mm
└─ Click Generate      →    on leading edge. Tolerance check passed."
                            ↓
                       Stage 1 preprocessing will correct:
                       "Installation completed in Bay 7.
                        Blade Gamma measurement: 3.02 mm..."
```

**Pro Tip:** You can also **manually edit** the generated text in Stage 1 before preprocessing! This lets you test specific scenarios.

##### **Step 2: Stage 1 - Data Preprocessing**

**You have THREE options for data input:**

#### **Option A: Use Generated Test Fact (Recommended for Testing)**

1. **After clicking "Generate Test Fact"** in sidebar, go to Verification Process tab
2. **Expand "📊 Configure & Run Preprocessing"**
3. **See the generated text** already filled in the "📝 Raw Text Input" field
4. **Domain, Location, and Timestamp** are pre-filled based on your selections
5. **Click "▶️ Run Stage 1 Preprocessing"**

#### **Option B: Upload Your Own File (For Real Data)**

1. **Expand "📊 Configure & Run Preprocessing"**
2. **Click "Browse files" under "📁 Upload Raw Data File"**
3. **Select your file**:
   - **TXT files**: Plain text (inspection reports, clinical notes, incident reports)
   - **JSON files**: Structured data (see format below)
   - **PDF files**: Document scans (*requires PyPDF2, currently placeholder only)

4. **Select Domain** matching your file content
5. **Set Location and Timestamp** metadata
6. **Click "▶️ Run Stage 1 Preprocessing"**

**Supported File Formats:**

**TXT Format (Plain Text):**
```
Any plain text file containing:
- Inspection reports
- Clinical notes
- Incident narratives
- Assembly documentation

Example aerospace_inspection.txt:
Installation completed in bay 7. Blade Gamma measurement: 3.02 mm 
on leading edge. Tolerance check passed.
```

**JSON Format (Structured):**
```json
{
  "document_id": "inspection_report_001",
  "domain": "aerospace",
  "format": "text",
  "content": {
    "text": "Installation completed in bay 7. Blade Gamma measurement: 3.02 mm"
  },
  "metadata": {
    "timestamp": "2024-10-29T10:30:00Z",
    "location": "Bay 7",
    "source_type": "inspection_report"
  }
}
```

**JSON Array Format (Multiple Documents):**
```json
[
  {
    "document_id": "doc_001",
    "domain": "aerospace",
    "content": {"text": "First inspection..."},
    "metadata": {"timestamp": "2024-10-29T10:00:00Z", "location": "Bay 3"}
  },
  {
    "document_id": "doc_002",
    "domain": "aerospace",
    "content": {"text": "Second inspection..."},
    "metadata": {"timestamp": "2024-10-29T11:00:00Z", "location": "Bay 7"}
  }
]
```

**What Happens When You Upload:**
- ✅ File content is read and parsed
- ✅ For TXT: Text extracted directly
- ✅ For JSON: Document structure preserved
- ✅ For PDF: Placeholder extraction (will show "[PDF extraction placeholder]")
- ✅ Domain and metadata used for preprocessing
- ✅ Your file name shown in success message

#### **Option C: Manually Enter/Edit Text**

1. **Expand "📊 Configure & Run Preprocessing"**
2. **Clear or edit the "📝 Raw Text Input" field**
3. **Type or paste your own text**:
   - Inspection reports
   - Clinical notes
   - Incident descriptions
   - Any domain-specific text

4. **Set Domain, Location, and Timestamp** manually
5. **Click "▶️ Run Stage 1 Preprocessing"**

**Use Cases:**
- Testing specific edge cases
- Tweaking generated text
- Processing custom scenarios
- Testing preprocessing on unusual text

---

#### **After Running Stage 1 - What You'll See:**

**Preprocessing Results Display:**

**Left Side (BEFORE):**
- 📥 Original raw text
- Metadata: domain, location, timestamp

**Right Side (AFTER):**
- 📤 Normalized text (spell-corrected, terminology standardized)
- Spatiotemporal: timestamp_utc, coordinates, location_symbol

**Preprocessing Changes Applied:**
- ✓ Spell corrections
- ✓ Terminology standardization
- ✓ Temporal alignment to UTC
- ✓ Spatial mapping (symbolic → coordinates)

**Status Indicator:**
- 🟢 Green box: "Stage 1 Complete" → Ready for Stage 2
- 🔴 Red box: "Stage 1 Not Completed" → Run preprocessing first

3. **Click "▶️ Run Stage 1 Preprocessing"**

**What Happens:**
- Spell correction: "measurment" → "measurement"
- Terminology standardization: "bay 7" → "Bay 7", "micu" → "MICU"
- Temporal alignment: Local time → UTC ISO 8601
- Spatial mapping: "Bay 7" → (40.0, 20.0, 0.0)

**What You'll See:**
- ✅ Before/After comparison (raw vs normalized text)
- ✅ List of preprocessing changes applied
- ✅ Green status indicator when complete

##### **Step 3: Stage 2 - LLM Extraction**

1. **Wait for Stage 1 to complete** (green indicator)
2. **Click "▶️ Run Stage 2 LLM Extraction"**

**What Happens:**
- Domain-specialized prompt applied (view in "View Prompt Template" expander)
- Facts extracted from preprocessed text
- LLM confidence assigned based on text quality:
  - Complete info → 0.9 (high confidence)
  - Partial info → 0.8 (medium confidence)
  - Minimal info → 0.6 (low confidence)

**What You'll See:**
- ✅ Extracted candidate facts with triple structure (s, r, o)
- ✅ LLM confidence levels color-coded
- ✅ Spatiotemporal coordinates
- ✅ Green status indicator when complete

##### **Step 4: Stage 3 - TruthFlow Verification**

1. **Wait for Stage 2 to complete** (or use demo facts)
2. **Click "▶️ Run TruthFlow Verification"**

**What Happens:**
- **5 Verification Modules** execute sequentially:
  1. LOV (Lexical-Ontological Verification) - Ontology compliance
  2. POV (Protocol-Ontology Verification) - Standards compliance
  3. MAV (Motion-Aware Verification) - Physics constraints (ψ_s, ψ_t, Ψ)
  4. WSV (Web-Source Verification) - External corroboration
  5. ESV (Embedding Similarity Verification) - Statistical anomaly detection

- **Module Activation**: Each module contributes if S_i ≥ θ_i
- **Early Termination**: Stops when cumulative confidence C ≥ Θ
- **Decision Logic**:
  - Accept: C ≥ 0.65
  - Review: 0.55 ≤ C < 0.65
  - Reject: C < 0.55

**What You'll See (Right Column):**
- ✅ Decision: Accept/Review/Reject (color-coded)
- ✅ Cumulative confidence calculation breakdown
- ✅ STKG integration status (green = added, red = rejected, orange = review)
- ✅ Module performance gauges
- ✅ RMMVe execution details
- ✅ Early termination notification (if applicable)
- ✅ Module confidence vs thresholds chart

---

### Tab 4: 🔄 AAIC Monitoring

**What You'll See:**
- CGR-CUSUM cumulative sums for each module
- Performance shift detection
- Parameter adjustment triggers
- Historical adaptation events

**What It's For:**
Monitoring how the system adapts to distribution drift and performance changes.

---

### Tab 5: 📊 Parameter Evolution

**What You'll See:**
- Weight evolution over time
- Threshold adjustments
- Alpha value changes
- Comparison: current vs initial parameters

**What It's For:**
Understanding how AAIC adapts module parameters based on performance.

---

### Tab 6: 📜 Verification History

**What You'll See:**
- Complete list of all verified facts
- Performance trends
- Quality distribution
- Decision breakdown (Accept/Review/Reject counts)

**What It's For:**
Audit trail and historical performance analysis.

---

## Testing Workflows

### **Workflow 1: Quick Demo (Pre-generated Facts)**

**Use Case:** Test the system quickly without uploading files

**Steps:**
1. Open **Verification Process** tab
2. Sidebar: Select quality level (e.g., "high_quality")
3. Click "🎲 Generate Test Fact" (generates demo fact)
4. Click "▶️ Run TruthFlow Verification" in Stage 3
5. See results in right column
6. Go to **STKG Structure** tab → See fact added to knowledge graph (if accepted)

**Expected Time:** 30 seconds

---

### **Workflow 2: Complete Pipeline (Upload Real Data)**

**Use Case:** Test with your own inspection reports, clinical notes, etc.

**Steps:**
1. Open **Verification Process** tab
2. **Stage 1**:
   - Click "Upload Raw Data File" or paste text manually
   - Select domain (aerospace/healthcare/aviation/cad)
   - Set location and timestamp
   - Click "▶️ Run Stage 1 Preprocessing"
   - Review before/after comparison
3. **Stage 2**:
   - Click "▶️ Run Stage 2 LLM Extraction"
   - Review extracted facts with confidence scores
4. **Stage 3**:
   - Click "▶️ Run TruthFlow Verification"
   - Review decision and confidence breakdown
5. **STKG Structure** tab:
   - See accepted facts integrated into knowledge graph

**Expected Time:** 2-3 minutes

---

### **Workflow 3: Quality Comparison**

**Use Case:** See how different quality levels affect verification

**Steps:**
1. Sidebar: Select "high_quality"
2. Click "🎲 Generate Test Fact"
3. Run all three stages → Should see "Accept" decision
4. Go to **STKG Structure** → See fact added ✅
5. Sidebar: Select "low_quality"
6. Click "🎲 Generate Test Fact"
7. Run all three stages → Should see "Reject" decision
8. Go to **STKG Structure** → Recent updates unchanged (fact not added) ❌

**Expected Time:** 1-2 minutes

---

### **Workflow 4: Batch Processing (Automated Testing)**

**Use Case:** Generate and verify multiple facts automatically to test AAIC adaptation and accumulate verification history

**What It Does:**
- Automatically generates 10 facts (or custom number)
- Runs complete 3-stage pipeline for each fact
- Randomly selects quality levels (high/medium/low)
- Uses your selected domain from sidebar
- Records all verifications in history
- Triggers AAIC parameter adaptation
- Integrates accepted facts into STKG

**Steps:**
1. **Sidebar → Batch Processing section**
2. **Select number of facts** (default: 10)
3. **(Optional) Check "Force Performance Shifts"** to guarantee some AAIC adaptations
4. **Click "Process 10 Facts" button**

**What Happens:**
- Progress bar shows processing status
- Each fact goes through:
  - ✅ **Stage 1**: Raw text generation → Preprocessing
  - ✅ **Stage 2**: LLM extraction (real API or simulation)
  - ✅ **Stage 3**: TruthFlow verification
- AAIC monitors each verification and adapts parameters
- All facts recorded in verification history

**Results Summary (appears after completion):**
```
✅ Batch completed: 7 Accepted, 2 Review, 1 Rejected
🔄 3 performance shifts detected and parameters adjusted by AAIC
📊 7 facts integrated into STKG
```

**Where to View Results:**
- **📜 Verification History** tab → See all 10 verifications with decisions
- **🔄 AAIC Monitoring** tab → View CGR-CUSUM and detected shifts
- **📊 Parameter Evolution** tab → Track weight/threshold changes
- **🗂️ STKG Structure** tab → See accepted facts in knowledge graph

**Expected Time:** 30-60 seconds (depending on LLM mode)

**💡 Tips:**
- Use **demo mode** for fast batch processing
- Use **real API** for production-quality extraction
- Enable **"Force Performance Shifts"** to see AAIC in action
- Check **terminal/console output** for debug information

---

### **Workflow 5: AAIC Adaptation Testing**

**Use Case:** See how the system adapts to performance shifts

**Steps:**
1. Sidebar: Enable "Introduce Performance Shifts"
2. Select "LOV" from shift module dropdown
3. Generate and verify 5-10 facts
4. Go to **AAIC Monitoring** tab
5. See cumulative sums increasing
6. When threshold exceeded → See parameter adjustments
7. Go to **Parameter Evolution** tab
8. See weight/threshold/alpha changes over time

**Expected Time:** 3-5 minutes

---

## Understanding Results

### **Decision Outcomes**

#### ✅ **Accept (Green)**
- **Meaning**: Fact passed verification with high confidence (C ≥ 0.65)
- **Action**: Automatically added to STKG
- **What to do**: View in STKG Structure tab to see integration

#### 🔍 **Review (Orange)**
- **Meaning**: Borderline confidence (0.55 ≤ C < 0.65)
- **Action**: Requires human expert review
- **What to do**: Manually inspect fact details before approving

#### ❌ **Reject (Red)**
- **Meaning**: Failed verification (C < 0.55)
- **Action**: Not added to STKG
- **What to do**: Check module details to see why it failed

### **Cumulative Confidence Formula**

```
C = Σ(wᵢ × Sᵢ) / Σ(wᵢ)
```

Where:
- **C**: Cumulative confidence
- **wᵢ**: Module trust weight
- **Sᵢ**: Module score = conf_k × [αᵢ · Metric1 + (1-αᵢ) · Metric2]
- **Σ**: Sum over activated modules only (where Sᵢ ≥ θᵢ)

The dashboard shows this calculation step-by-step!

### **Module Performance**

Each gauge shows:
- **Confidence score**: How confident this module is
- **Threshold**: Minimum score for activation
- **Status**: Activated (contributes) or Not Activated

### **Early Termination**

If you see a **green "Early Termination Achieved!"** box:
- ✅ System stopped early (high confidence reached)
- ✅ Saved computational cost by skipping expensive modules
- ✅ Still accurate (confidence already high enough)

---

## Advanced Features

### **Upload Custom Files**

**Supported Formats:**
- **TXT**: Plain text (inspection reports, clinical notes)
- **JSON**: Structured data with domain, content, metadata
- **PDF**: Placeholder support (requires PyPDF2 for full extraction)

**JSON Format Example:**
```json
{
  "document_id": "my_doc_001",
  "domain": "aerospace",
  "format": "text",
  "content": {
    "text": "Blade Gamma measurement: 3.02 mm at Bay 7"
  },
  "metadata": {
    "timestamp": "2024-10-29T10:30:00Z",
    "location": "Bay 7"
  }
}
```

### **View Prompt Templates**

In Stage 2, expand "📝 View Prompt Template (Listing 1)" to see:
- Domain-specialized prompts
- Schema constraints from ontology
- Few-shot examples
- Output structure requirements

This is the actual prompt sent to the LLM for fact extraction!

### **Monitor AAIC Adaptation**

1. Enable "Introduce Performance Shifts" in sidebar
2. Verify multiple facts (10+)
3. Watch AAIC Monitoring tab for shift detection
4. See Parameter Evolution tab for adaptation history

### **Export Results**

Verification results are stored in session state and can be exported via:
- **Verification History** tab → Shows all verifications
- **STKG Structure** tab → Shows accepted facts

---

## Common Questions

### **Q: Why was my high-quality fact rejected?**

**A:** Check the verification results:
- Was LLM confidence actually high? (Should be 0.9-1.0)
- Did modules activate? (Check cumulative confidence calculation)
- Was there a performance shift introduced? (Check sidebar settings)

The system is **honest** - if preprocessing produced poor normalized text or extraction failed to capture complete information, confidence will be lower.

### **Q: How do I test different domains?**

**A:** 
1. Sidebar: Select domain (aerospace/healthcare/aviation/cad)
2. Click "Generate Test Fact"
3. Stage 1 will use domain-specific:
   - Terminology (Bay 7 vs MICU vs Runway 24L)
   - Facility coordinates
   - Ontology classes
   - Physical constraints

### **Q: What if I don't have an OpenAI API key?**

**A:** No problem! The system falls back to **simulation mode**:
- Parses text using regex to extract values
- Assigns confidence based on text completeness
- Still demonstrates the complete pipeline honestly

To use real LLM:
```bash
export OPENAI_API_KEY="your-key-here"
streamlit run app.py
```

### **Q: How can I see what's in the knowledge graph?**

**A:** Go to **STKG Structure** tab:
- See entity count, relationship count
- View recent additions (last 5 accepted facts)
- Browse ontology structure
- See domain-specific examples

### **Q: What does "Early Termination" mean?**

**A:** The verification stopped early because:
- Cumulative confidence already ≥ global threshold (0.65)
- No need to run expensive modules (WSV, ESV)
- Saves ~40% computation time
- Still accurate (confidence is already sufficient)

---

## Testing Scenarios

### **Scenario 1: Perfect Aerospace Measurement**

```
Domain: Aerospace
Quality: High Quality
Expected: ACCEPT ✅

Sample Text:
"Installation completed in Bay 7. Blade Alpha measurement: 3.01 mm 
on leading edge. Nominal: 3.0 mm. Deviation: 0.01 mm. 
Tolerance: ±0.1 mm. Status: PASS."

Expected Results:
- Stage 1: Spell corrections, Bay 7 → (40.0, 20.0, 0.0)
- Stage 2: LLM confidence 0.9 (high)
- Stage 3: Accept (C ≈ 0.85), Added to STKG ✓
```

### **Scenario 2: Vague Healthcare Note**

```
Domain: Healthcare
Quality: Low Quality
Expected: REJECT ❌

Sample Text:
"Patient transferred. Went to another unit."

Expected Results:
- Stage 1: Minimal normalization (not much to correct)
- Stage 2: LLM confidence 0.3 (very low) - missing patient ID, units, times
- Stage 3: Reject (C ≈ 0.0), Not added to STKG
```

### **Scenario 3: Spelling Errors with Good Data**

```
Domain: Aerospace
Quality: Medium Quality
Expected: ACCEPT ✅

Sample Text:
"Instalation completed in bay 7. Blade Gamma measurment: 3.02 mm."

Expected Results:
- Stage 1: Corrections applied (Instalation→Installation, measurment→measurement)
- Stage 2: LLM confidence 0.8-0.9 (good data despite spelling)
- Stage 3: Accept (C ≈ 0.84), Added to STKG ✓
```

---

## Tips for Best Results

### ✅ **DO:**
- Start with high-quality examples to understand the system
- Use the domain-appropriate terminology
- Include specific measurements, locations, and timestamps
- Review the before/after preprocessing to see what changed
- Check the STKG Structure tab after each verification

### ❌ **DON'T:**
- Don't expect low-quality facts to be accepted (system is honest!)
- Don't skip Stage 1 preprocessing (required for Stage 2)
- Don't expect instant results with API calls (simulation is faster)
- Don't be surprised if borderline facts get "Review" status

---

## Example Testing Session

### **15-Minute Complete Demo**

**Minutes 0-3: Explore Tabs**
- Tab 1 (Methodology): Read STKG formalization
- Tab 2 (STKG Structure): See empty knowledge graph (0 entities, 0 relationships)

**Minutes 3-6: Test High Quality Fact**
- Sidebar: Domain=Aerospace, Quality=High
- Generate Test Fact
- Run Stage 1 → See preprocessing corrections
- Run Stage 2 → See LLM confidence 0.9
- Run Stage 3 → See ACCEPT decision
- Tab 2 (STKG): See fact added! (1 entity, 1 relationship)

**Minutes 6-9: Test Low Quality Fact**
- Sidebar: Quality=Low
- Generate Test Fact
- Run all stages → See REJECT decision
- Tab 2 (STKG): No change (fact not added)

**Minutes 9-12: Upload Real File**
- Stage 1: Upload your inspection report/clinical note
- Process through pipeline
- See honest verification results

**Minutes 12-15: Review Results**
- Tab 6 (History): See all verifications
- Tab 4 (AAIC): Check if any shifts detected
- Tab 5 (Parameters): See if weights/thresholds adapted

---

## Troubleshooting

### **Problem: "Stage 2 Blocked" message**

**Solution:** Complete Stage 1 preprocessing first. Click "Run Stage 1 Preprocessing" button.

### **Problem: "No facts available" in Stage 3**

**Solution:** Either:
1. Complete Stage 2 LLM extraction, OR
2. Generate a test fact using sidebar controls

### **Problem: Results appear in wrong column**

**Solution:** Refresh the page (Ctrl+R or Cmd+R). Results should appear in right column only after clicking "Run TruthFlow Verification".

### **Problem: All facts get rejected**

**Solution:** Check:
- Is quality level set to "low_quality"? (Expected behavior)
- Is preprocessing producing good normalized text?
- Is LLM confidence very low (<0.5)?

The system is **honest** - poor quality input → low confidence → rejection.

---

## Additional Resources

- **Experiments Guide**: `experiments/EXPERIMENTS_GUIDE.md`
- **Quick Start**: `QUICKSTART.md`
- **README**: `README.md`
- **Implementation Summary**: `FINAL_IMPLEMENTATION_SUMMARY.md`
- **Verification Process**: `VERIFICATION_PROCESS_COMPLETE.md`

---

## Support

For questions or issues:
1. Check the methodology tab for theoretical background
2. Review the STKG Structure tab to understand the knowledge graph
3. Use the verification history tab to debug failed verifications
4. Check the console/terminal for error messages

---

## Summary

ATLASky-AI provides an **interactive demonstration** of spatiotemporal knowledge graph verification:

✅ **Three-stage pipeline**: Preprocessing → LLM Extraction → TruthFlow Verification  
✅ **Multi-domain support**: Aerospace, healthcare, aviation, CAD  
✅ **Honest verification**: Real quality assessment, no fake scores  
✅ **Automatic STKG integration**: Accepted facts added to knowledge graph  
✅ **Adaptive parameters**: AAIC learns and adjusts over time  

**Start exploring now by selecting a domain and quality level, then clicking "Generate Test Fact"!** 🚀

