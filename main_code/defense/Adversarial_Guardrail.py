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
        self.th_comment_hard = args.th_comment_hard
        self.th_comment_grey = args.th_comment_grey
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

    def regenerate_comment(self, full_code, comment_start_byte):
        prefix_code = full_code[:comment_start_byte].decode("utf8", errors='ignore')
        context_window = prefix_code[-512:] 
        
        if not context_window.strip().endswith(("//", "/*")):
            prompt = context_window + "//"
        else:
            prompt = context_window

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("\n")[0]
            )
        generated_text = self.tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        clean_comment = generated_text.split('\n')[0].strip()
        return f"/* {clean_comment} */"

    def detect(self, code):
        """
        執行 Layer 1 & 2 的檢測與修復
        Returns: l1_triggered (bool), l2_triggered (bool), repaired_code (str)
        """
        if not code or not self.language: return False, False, code
        
        code_bytes = bytes(code, "utf8")
        try:
            tree = self.parser.parse(code_bytes)
        except:
            return False, False, code

        query = self.language.query("""
            (comment) @comment
            (string_literal) @string
        """)
        captures = query.captures(tree.root_node)
        
        replacements = [] 
        l1_triggered = False
        l2_triggered = False
        
        for node, type_name in captures:
            text = node.text.decode("utf8", errors='ignore')
            start = node.start_byte
            end = node.end_byte
            
            if len(text) < 6: continue 

            score = self.calc_mink_score(text, k=0.2)
            
            if type_name == 'comment':
                if score > self.th_comment_hard:
                    replacements.append((start, end, "/* garbage removed */"))
                    l1_triggered = True
                elif score > self.th_comment_grey:
                    new_comment = self.regenerate_comment(code_bytes, start)
                    replacements.append((start, end, new_comment))
                    l2_triggered = True
                
            elif type_name == 'string':
                if score > self.th_string_hard:
                    replacements.append((start, end, '""')) 
                    l1_triggered = True
        
        if not replacements:
            return False, False, code

        replacements.sort(key=lambda x: x[0], reverse=True)
        new_code_bytes = list(code_bytes)
        for start, end, rep_text in replacements:
            rep_bytes = bytes(rep_text, "utf8")
            new_code_bytes[start:end] = rep_bytes
            
        return l1_triggered, l2_triggered, bytes(new_code_bytes).decode("utf8", errors='ignore')