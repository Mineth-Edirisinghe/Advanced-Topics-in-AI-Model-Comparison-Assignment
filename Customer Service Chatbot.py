#Qwen 3.5 4B vs. Ministral 3 3B

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers import(
    Mistral3ForConditionalGeneration, 
    MistralCommonBackend,
    FineGrainedFP8Config
)

def qwen():
    model_id = "Qwen/Qwen3.5-4B"
    project_folder = Path(__file__).resolve().parent

    company_information = (
        project_folder / "company_information.txt"
    ).read_text(encoding="utf-8")

    output_folder = project_folder / "outputs" / "customer_service_chatbot"
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"qwen_chat_{timestamp}.json"

    print("Loading Qwen")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    system_message = {
        "role": "system",
        "content": (
            "You are BlueScape Pool Builder's customer service assistant."
            "Use the company information below as your factual source. "
            "If information is missing, say our team must confirm it. "
            "Keep responses friendly, concise and relevant. "
            "Do not claim to submit enquiries or book appointments.\n\n"
            + company_information
        )
    }

    messages = [system_message]
    conversation_id = 1

    results = {
        "model": model_id,
        "started_at": timestamp,
        "company_information": company_information,
        "settings": {
            "max_new_tokens": 512,
            "do_sample": False,
            "enable_thinking": False
        },
        "exchanges": []
    }

    print("\nBlueScape chatbot ready.")
    print("Type /reset for a fresh conversation, or exit to stop.")
    print(f"Answers will be saved to: {output_file}\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "/reset":
            messages = [system_message]
            conversation_id += 1
            print("Conversation reset.\n")
            continue

        if not question:
            continue

        messages.append({"role": "user", "content": question})

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        response_seconds = time.perf_counter() - start

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        messages.append({"role": "assistant", "content": answer})

        print(f"\nBlueScape: {answer}")
        print(f"Response time: {response_seconds:.2f} seconds\n")

        results["exchanges"].append({
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "response_seconds": round(response_seconds, 3),
            "generated_tokens": len(generated_tokens),
            "token_limit_reached": len(generated_tokens) >= 512
        })

        output_file.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print("Chat ended.")



def ministral():
    model_id = "mistralai/Ministral-3-3B-Instruct-2512"
    project_folder = Path(__file__).resolve().parent
    
    company_information = (
            project_folder / "company_information.txt"
    ).read_text(encoding="utf-8")
    
    output_folder = project_folder / "outputs" / "customer_service_chatbot"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"ministral_chat_{timestamp}.json"
    
    print("Loading Ministral")
    
    processor = MistralCommonBackend.from_pretrained(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True)
    )
    model.eval()
    
    system_message = {
        "role": "system",
        "content": (
            "You are BlueScape Pools' customer-service assistant. "
            "Use the company information below as your factual source. "
            "If information is missing, say our team must confirm it. "
            "Keep responses friendly, concise and relevant. "
            "Do not claim to submit enquiries or book appointments.\n\n"
            + company_information
        )
    }

    messages = [system_message]
    conversation_id = 1

    results = {
        "model": model_id,
        "started_at": timestamp,
        "company_information": company_information,
        "settings": {
            "max_new_tokens": 512,
            "do_sample": False,
            "enable_thinking": False
        },
        "exchanges": []
    }

    print("\nBlueScape chatbot ready.")
    print("Type /reset for a fresh conversation, or exit to stop.")
    print(f"Answers will be saved to: {output_file}\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if question.lower() == "/reset":
            messages = [system_message]
            conversation_id += 1
            print("Conversation reset.\n")
            continue

        if not question:
            continue

        messages.append({"role": "user", "content": question})

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        response_seconds = time.perf_counter() - start

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        messages.append({"role": "assistant", "content": answer})

        print(f"\nBlueScape: {answer}")
        print(f"Response time: {response_seconds:.2f} seconds\n")

        results["exchanges"].append({
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "response_seconds": round(response_seconds, 3),
            "generated_tokens": len(generated_tokens),
            "token_limit_reached": len(generated_tokens) >= 512
        })

        output_file.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print("Chat ended.")





if __name__ == "__main__":
    ministral()