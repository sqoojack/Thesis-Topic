import torch
import numpy as np

class AdversarialGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        
        # 接收 main.py 傳來的參數
        # 移除 Grey/Hard 區分，統一使用 adversarial_threshold
        self.adversarial_threshold = args.adversarial_threshold
        self.th_string_hard = args.th_string_hard

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.cpu().numpy()

    def calc_mink_score(self, text, k=0.2):
        if not text or len(text) < 4: return 0.0
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] <= 2: return 0.0
        losses = self.get_token_losses(inputs["input_ids"])
        if len(losses) == 0: return 0.0
        sorted_losses = np.sort(losses)[::-1]
        num_tokens = max(1, int(len(losses) * k))
        top_k_losses = sorted_losses[:num_tokens]
        return np.mean(top_k_losses)

    def detect(self, code):
        """
        執行 Adversarial 檢測與修復 (直接刪除惡意部分)
        Returns: triggered (bool), repaired_code (str)
        """
        if not code or not self.language: return False, code
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except:
            return False, code

        query = self.language.query("""
            (comment) @comment
            (string_literal) @string
        """)
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        triggered = False
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            start = node.start_byte
            end = node.end_byte
            
            if len(text) < 6: continue 

            score = self.calc_mink_score(text, k=0.2)
            
            if type_name == 'comment':
                # 只要超過閾值，直接刪除 (替換為空字串)
                if score > self.adversarial_threshold:
                    replacements.append((start, end, "")) 
                    triggered = True
                
            elif type_name == 'string':
                # 字串通常 entropy 較高，建議保留獨立的閾值，或者也可以統一
                if score > self.th_string_hard:
                    replacements.append((start, end, '""')) 
                    triggered = True
        
        if not replacements:
            return False, code

        # 執行替換
        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = list(code_bytes)
        for start, end, rep_text in replacements:
            rep_bytes = bytes(rep_text, "utf8")
            new_code_bytes[start:end] = rep_bytes
            
        return triggered, bytes(new_code_bytes).decode("utf8", errors='ignore')