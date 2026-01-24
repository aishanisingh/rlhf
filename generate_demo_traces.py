"""
Generate demo traces for testing the RLHF Workbench.

Since small base models like distilgpt2 don't follow instructions,
this script creates synthetic traces that demonstrate the system's
optimization logic with realistic medical reasoning patterns.

For production use, switch to an instruction-tuned model like:
- microsoft/phi-2
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Or use OpenAI/Groq API
"""
import json
import os
from pathlib import Path

from config import get_config
from sampler import load_all_samples, TraceCache, compute_normalized_hash


# Pre-written trace templates that vary in reasoning quality
TRACE_TEMPLATES = {
    "sample1": {
        # Answer: B (Patent Foramen Ovale)
        "traces": [
            {
                "text": """Let me analyze this case systematically. The patient presents with sudden weakness in the left arm and leg, which suggests a neurological event - likely a stroke affecting the right side of the brain.

Key observation: The patient has a swollen, tender right lower leg after long-distance travel. This is classic for deep vein thrombosis (DVT).

Now, the critical question is: how could a venous clot cause an arterial stroke? Normally, venous blood goes through the lungs where clots would be filtered. However, if there's a right-to-left cardiac shunt, a clot could bypass the lungs entirely.

A Patent Foramen Ovale (PFO) is exactly such a shunt - a hole between the right and left atria that persists from fetal circulation. It's present in about 25% of adults.

This explains the paradoxical embolism: DVT clot → right atrium → through PFO → left atrium → brain → stroke.

The answer is B.""",
                "quality": "best"
            },
            {
                "text": """This patient has neurological symptoms (left-sided weakness) combined with signs of DVT (swollen leg after travel).

When we see stroke symptoms in a young patient with DVT, we should think about paradoxical embolism. This happens when a venous clot crosses to the arterial circulation through a cardiac defect.

Looking at the options:
- Mitral Valve Prolapse: causes regurgitation, not shunting
- Patent Foramen Ovale: allows right-to-left shunting - this fits!
- Hypertrophic Cardiomyopathy: causes outflow obstruction
- VSD: usually left-to-right shunt in adults

The answer is B.""",
                "quality": "middle"
            },
            {
                "text": """The patient has weakness on one side and a swollen leg. The leg swelling after travel suggests a clot.

I need to find a heart problem that connects these. A hole in the heart could let a clot go to the brain.

Patent Foramen Ovale is a hole between atria.

The answer is B.""",
                "quality": "worst"
            }
        ]
    },
    "sample2": {
        # Answer: B (Lower lobe of left lung)
        "traces": [
            {
                "text": """This is a trauma case requiring careful anatomical analysis. Let me work through this systematically.

The wound is at the upper border of the 8th rib in the left midaxillary line, with 5cm depth. I need to consider what structures lie in this region.

Anatomical considerations:
- The 8th rib at the midaxillary line is at approximately the level of T8-T9
- This is below the cardiac silhouette but still within the thoracic cavity
- The left hemidiaphragm attaches around ribs 7-10 laterally
- The lower lobe of the left lung extends to about the 10th rib in the midaxillary line

Given the location and depth:
- The heart (left atrium) is too medial and superior
- The spleen is more posterior and inferior
- The liver is on the right side
- The lower lobe of the left lung is directly in the path

The patient's vitals (tachycardia, hypotension) suggest hemothorax or pneumothorax, consistent with lung injury.

The answer is B.""",
                "quality": "best"
            },
            {
                "text": """A stab wound at the 8th rib, left midaxillary line, 5cm deep. The patient is hypotensive and tachycardic.

At this level on the left side, the main structures are:
- Lower lobe of left lung
- Possibly the diaphragm
- The spleen is nearby but more posterior

The clinical picture with unstable vitals suggests bleeding into the chest, most likely from the lung.

The left atrium is too high and medial. The liver is on the wrong side.

The answer is B.""",
                "quality": "middle"
            },
            {
                "text": """Stab wound on the left side of the chest at rib 8. The patient's blood pressure is low.

At this location, the lung would be hit. The lower lobe of the left lung is in this area.

The answer is B.""",
                "quality": "worst"
            }
        ]
    },
    "sample3": {
        # Answer: C (Chromosome 9)
        "traces": [
            {
                "text": """The clinical presentation of progressive gait disturbances, tremors, and speech difficulties points toward a neurodegenerative condition affecting the cerebellum and its connections.

The genetic finding of GAA trinucleotide repeat expansions is the definitive clue. This pattern is pathognomonic for Friedreich's ataxia.

Let me recall the genetics:
- Friedreich's ataxia is caused by mutations in the FXN gene
- The FXN gene encodes frataxin, a mitochondrial protein
- The gene is located on chromosome 9q21

Let me verify by excluding other options:
- Chromosome 4: Huntington disease (CAG repeats) - different presentation
- Chromosome 6: Not associated with major trinucleotide repeat disorders
- Chromosome X: Fragile X syndrome (CGG repeats) - different presentation

The FXN gene on chromosome 9 is the correct answer.

The answer is C.""",
                "quality": "best"
            },
            {
                "text": """GAA trinucleotide repeats are characteristic of Friedreich's ataxia. This autosomal recessive condition affects the nervous system and heart.

The responsible gene is FXN (frataxin), and I need to remember which chromosome it's on.

Friedreich's ataxia is associated with chromosome 9 - specifically 9q21 where the FXN gene is located.

The other chromosomes listed are associated with different conditions (Huntington's is on 4, Fragile X involves the X chromosome).

The answer is C.""",
                "quality": "middle"
            },
            {
                "text": """GAA repeats cause Friedreich's ataxia. The gene for this is on chromosome 9.

The answer is C.""",
                "quality": "worst"
            }
        ]
    },
    "sample4": {
        # Answer: C (Oral ciprofloxacin)
        "traces": [
            {
                "text": """This is a critically ill patient with fever, hypotension, and severely abnormal labs. Let me analyze systematically.

Key findings:
- Severe anemia: Hb 5 g/dL (critical)
- Abnormal differential: 2% neutrophils (profound neutropenia), 96% lymphocytes
- Signs of sepsis/shock: fever, hypotension

The differential with only 2% polymorphs indicates severe neutropenia. Combined with fever and hypotension, this is febrile neutropenia with septic shock - a medical emergency.

Management priorities:
- IV fluid resuscitation: Essential for shock - should be given
- Packed RBC transfusion: Critical for Hb of 5 - should be given
- IV broad-spectrum antibiotics: Standard of care for febrile neutropenia - should be given

What should be AVOIDED?
- Oral ciprofloxacin: In septic shock, GI absorption is unreliable due to poor perfusion. IV antibiotics are mandatory. Oral route is inadequate for this acuity.

The answer is C.""",
                "quality": "best"
            },
            {
                "text": """This patient has febrile neutropenia (2% polymorphs) with signs of septic shock.

The treatment needs to be aggressive:
- Fluids for hypotension - yes
- Blood transfusion for severe anemia - yes
- Antibiotics - yes, but must be IV

Oral ciprofloxacin would be inappropriate because:
1. The patient is in shock with unreliable GI absorption
2. Febrile neutropenia requires IV antibiotics
3. Oral medications are too slow-acting for this emergency

The answer is C.""",
                "quality": "middle"
            },
            {
                "text": """The patient has low neutrophils and signs of infection with shock.

You can't give oral antibiotics to someone in septic shock. They need IV antibiotics.

The answer is C.""",
                "quality": "worst"
            }
        ]
    },
    "sample5": {
        # Answer: C (Ophthalmoplegia)
        "traces": [
            {
                "text": """This case requires careful neuroanatomical reasoning. Let me analyze the key features.

Clinical presentation:
- Severe headache in left forehead and eye, waking from sleep
- History of recent sinus infection
- Type 1 diabetes (increases infection risk)
- Imaging shows sinus thrombosis ABOVE the sella turcica

The location "above the sella turcica" is critical - this describes the cavernous sinus. Cavernous sinus thrombosis (CST) is a serious complication of facial/sinus infections.

Structures within the cavernous sinus:
- CN III (oculomotor)
- CN IV (trochlear)
- CN VI (abducens)
- CN V1 and V2 (trigeminal branches)
- Internal carotid artery

When these cranial nerves are affected by inflammation/thrombosis, the result is ophthalmoplegia - inability to move the eye properly.

Let me exclude other options:
- Anosmia: CN I is not in cavernous sinus
- Mandibular pain: V3 is not in cavernous sinus
- Vertigo: CN VIII is not in cavernous sinus
- Vision loss: Optic nerve is not in cavernous sinus

The answer is C.""",
                "quality": "best"
            },
            {
                "text": """The imaging shows thrombosis above the sella turcica - this is the cavernous sinus.

The cavernous sinus contains cranial nerves III, IV, VI (eye movement) and branches of V.

With cavernous sinus thrombosis, these nerves get compressed, causing ophthalmoplegia (paralysis of eye movements).

The olfactory nerve and vestibular system are not involved. The optic nerve runs separately.

The answer is C.""",
                "quality": "middle"
            },
            {
                "text": """Thrombosis above sella turcica means cavernous sinus problem. Eye movement nerves are in the cavernous sinus.

The answer is C.""",
                "quality": "worst"
            }
        ]
    }
}


def generate_demo_traces():
    """Generate demo traces and save to cache."""
    config = get_config()
    samples = load_all_samples(config)
    cache = TraceCache(config.sampler.cache_dir)

    print("Generating demo traces for all samples...")

    for sample in samples:
        sample_id = sample.sample_id

        if sample_id not in TRACE_TEMPLATES:
            print(f"  {sample_id}: No template available, skipping")
            continue

        template = TRACE_TEMPLATES[sample_id]
        verified_traces = []

        for i, trace_data in enumerate(template["traces"]):
            trace = {
                "trace_id": f"{sample_id}_trace_{i}",
                "text": trace_data["text"].strip(),
                "extracted_answer": sample.answer,
                "is_correct": True,
                "normalized_hash": compute_normalized_hash(trace_data["text"]),
                "token_count": len(trace_data["text"].split())
            }
            verified_traces.append(trace)

        cache.save(
            sample_id=sample_id,
            verified_traces=verified_traces,
            candidates=[],
            attempts=3
        )
        print(f"  {sample_id}: Generated {len(verified_traces)} traces (answer: {sample.answer})")

    print("\nDemo traces saved to cache. You can now run: streamlit run app.py")


if __name__ == "__main__":
    generate_demo_traces()
