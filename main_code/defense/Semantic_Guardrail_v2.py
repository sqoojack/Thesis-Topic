import torch
import numpy as np
import re
from collections import defaultdict

class SemanticGuardrail:
    def __init__(self, model, tokenizer, device, parser, language, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.parser = parser
        self.language = language
        
        # 參數
        self.base_influence_th = args.l3_base_influence
        self.surprise_tolerance = args.l3_surprise_tolerance
        
        # 建議顯存安全批次大小 (可根據您的 GPU 調整，350M 模型建議 4-8)
        self.inference_batch_size = 4 

        self.whitelist = {
            'int', 'char', 'void', 'float', 'double', 'long', 'short', 'unsigned', 'signed',
            'struct', 'union', 'enum', 'static', 'const', 'volatile', 'register', 'auto',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue',
            'return', 'goto', 'sizeof', 'typedef', 'main', 'true', 'false', 'NULL',
            'include', 'define', 'undef', 'ifdef', 'ifndef', 'endif', 'pragma',
            'args', 'argv', 'argc', 'data', 'buffer', 'buf', 'count', 'idx', 'index', 'len', 'size',
            'start', 'end', 'min', 'max', 'ctx', 'context', 'out', 'in', 'ptr', 'value', 'val',
            'recv', 'send', 'read', 'write', 'open', 'close', 'self', 'this', 'user', 'password'
        }
        
        self.noise_prefixes = ('trace_', 'debug_', 'test_', 'assert_', 'sys_', 'standard_', 'std_', 'av_', 'ff_')
        self.noise_suffixes = ('_init', '_exit', '_free', '_alloc', '_create', '_destroy', 
                                '_tab', '_table', '_list', '_queue', '_desc', '_info', '_data', 
                                '_ops', '_cb', '_ctx', '_t', '_s', '_eq', '_ne', '_impl', '_handler')

    def is_noisy_variable(self, text):
        if text.startswith(self.noise_prefixes): return True
        if text.endswith(self.noise_suffixes): return True
        return False

    def get_token_type(self, code, node, text):
        if text.isupper(): return 'MACRO'
        end_byte = node.end_byte
        next_chars = code[end_byte:end_byte+10].strip()
        if next_chars.startswith('('): return 'FUNC'
        return 'NORMAL'

    def get_batch_loss(self, texts):
        """
        修正版：增加內部 Mini-batch 循環以防止 OOM
        """
        if not texts:
            return np.array([])

        all_mean_losses = []

        # 將大量文本切分為多個小批次處理
        for i in range(0, len(texts), self.inference_batch_size):
            batch_texts = texts[i : i + self.inference_batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses = token_losses.view(shift_labels.size())

                masked_losses = token_losses * shift_mask
                sum_losses = masked_losses.sum(dim=1)
                valid_counts = shift_mask.sum(dim=1)
                
                batch_mean_losses = sum_losses / valid_counts.clamp(min=1)
                all_mean_losses.append(batch_mean_losses.cpu().numpy())
            
            # 顯式清理快取 (選用，有助於穩定顯存)
            # torch.cuda.empty_cache()

        return np.concatenate(all_mean_losses)

    def detect(self, code):
        """
        執行 Layer 3 Active Verify (Parallel Mini-batch Version)
        """
        if not code: return False, code, []

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            query = self.language.query("(identifier) @v")
            captures = query.captures(tree.root_node)
        except:
            return False, code, []

        candidates = []
        texts_orig = []
        texts_neutral = []
        processed_ranges = set()

        for node, _ in captures:
            text = node.text.decode("utf8", errors='ignore')
            range_key = (node.start_byte, node.end_byte)
            if range_key in processed_ranges: continue
            processed_ranges.add(range_key)

            if len(text) < 4 or text in self.whitelist: continue
            
            token_type = self.get_token_type(code, node, text)
            is_noisy = self.is_noisy_variable(text)
            
            # 視窗擷取與截斷
            prefix = code[:node.start_byte]
            suffix = code[node.end_byte:]
            eval_suffix = suffix[:256] 
            if len(eval_suffix) < 10: continue

            window_prefix = prefix[-768:] 
            
            texts_orig.append(window_prefix + text + eval_suffix)
            texts_neutral.append(window_prefix + "VAR_0" + eval_suffix)

            candidates.append({
                'var': text,
                'type': token_type,
                'is_noisy': is_noisy
            })

        if not candidates:
            return False, code, []

        # 批次取得 Loss
        losses_orig = self.get_batch_loss(texts_orig)
        losses_neutral = self.get_batch_loss(texts_neutral)

        toxic_vars = []
        is_attack = False
        debug_info = []

        for i, cand in enumerate(candidates):
            influence = losses_orig[i] - losses_neutral[i]
            
            dynamic_threshold = self.base_influence_th
            if cand['is_noisy']: dynamic_threshold *= 0.8
            if cand['type'] in ('FUNC', 'MACRO'): dynamic_threshold *= 2.5
            
            triggered = False
            if influence > 0.02 and influence > dynamic_threshold:
                triggered = True
                if cand['var'] not in toxic_vars:
                    toxic_vars.append(cand['var'])
                is_attack = True
            
            if triggered or influence > 0.01:
                debug_info.append({
                    "var": cand['var'],
                    "influence": float(influence),
                    "threshold": float(dynamic_threshold),
                    "is_noisy": cand['is_noisy'],
                    "type": cand['type'],
                    "triggered": triggered
                })

        repaired_code = code
        if is_attack:
            toxic_vars.sort(key=len, reverse=True)
            for idx, toxic_var in enumerate(toxic_vars):
                repaired_code = re.sub(rf'\b{re.escape(toxic_var)}\b', f"VAR_{idx}", repaired_code)

        return is_attack, repaired_code, debug_info