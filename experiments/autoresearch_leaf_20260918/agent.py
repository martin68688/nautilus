"""One agent, literal memory tools, then executable candidate; no reviewers."""
import ast
import json
from pathlib import Path
from core import atomic_json


def propose(call, folder, contract_text, incumbent_code, history, bridge=None):
    compatibility=json.loads(Path(__file__).with_name('memory_compatibility_v2.json').read_text()) if bridge else None
    messages = [{'role': 'system', 'content': contract_text + '''
Return JSON only. Tools are JSON actions, not shell or filesystem access.
{"action":"query","terms":["short literal"],"granularity":"architecture"}
{"action":"read","keys":["key from query results"]}
{"action":"submit","change":"one concrete change","code":"complete train.py",
 "decisions":[{"key":"read key","action":"adopt|adapt|reject|defer","reason":"why",
 "code_evidence":"exact implementing code excerpt or empty if rejected"}]}
No history memory is available unless explicitly supplied. Query only when it is.
The Python harness alone executes code and decides keep/discard. Historical scores
are observations, never your current score. Keep architecture freedom. Read full
cards before using them. Account for limitations and failed/regressed trials.
When memory is enabled, every proposal MUST query and read at least one relevant
complete card before submission and decide on every read card. Reject or defer
with a reason is valid; adoption and any particular model are NOT mandatory.
'''}, {'role': 'user', 'content': json.dumps({'incumbent_code': incumbent_code,
      'own_run_history': history[-12:], 'memory_catalogue': bridge.catalogue() if bridge else [],
      'migration_compatibility': compatibility}, ensure_ascii=False)}]
    read_keys = []
    allowed_keys = set()
    for turn in range(12):
        atomic_json(folder / f'agent-request-{turn:02d}.json', messages)
        reply, usage = call(messages)
        atomic_json(folder / f'agent-response-{turn:02d}.json', {'reply': reply, 'usage': usage})
        messages.append({'role': 'assistant', 'content': json.dumps(reply, ensure_ascii=False)})
        action = reply.get('action')
        if action == 'query' and bridge is not None:
            keys = bridge.grep(reply['terms'], reply.get('granularity'))
            allowed_keys.update(keys)
            result = {'matches': keys}
        elif action == 'read' and bridge is not None:
            keys = reply['keys']
            if not isinstance(keys,list) or not 1 <= len(keys) <= 6:
                raise ValueError('read 1-6 relevant full cards per request')
            if not set(keys) <= allowed_keys:
                raise ValueError('read must follow a real query')
            read_keys = list(dict.fromkeys(read_keys + keys))
            result = bridge.read(keys)  # Full unmodified cards; never truncate.
            result['migration_compatibility'] = compatibility
        elif action == 'submit':
            code = reply['code']
            (folder/'train.py').write_text(code)
            decisions = reply.get('decisions', [])
            try:
                ast.parse(code)
                if bridge is not None and not read_keys:
                    raise ValueError('memory-enabled proposal requires query followed by reading at least one relevant complete card; reject/defer is allowed, no recipe is forced')
                if {d['key'] for d in decisions} != set(read_keys):
                    raise ValueError('every read card requires an explicit decision')
                for d in decisions:
                    if d['action'] in ('adopt', 'adapt'):
                        if not d.get('code_evidence') or d['code_evidence'] not in code:
                            raise ValueError('claimed adoption lacks literal code evidence: '+d['key']+
                                '; quote ONE exact contiguous substring from code, without ellipses or reformatted whitespace')
            except (ValueError,SyntaxError) as error:
                result={'submission_accepted':False,'validation_error':str(error),
                        'retry':'Resubmit complete code and decisions with this mechanical error corrected; no training has occurred.'}
                atomic_json(folder/f'submission-rejected-{turn:02d}.json',result)
                messages.append({'role':'user','content':json.dumps(result)})
                continue
            if bridge is not None:
                receipt=bridge.receipt(read_keys, decisions)
                receipt['migration_compatibility']=compatibility
                atomic_json(folder / 'MEMORY_DECISIONS.json', receipt)
            elif decisions:
                raise ValueError('no-memory arm cannot claim historical decisions')
            return code, reply['change']
        else:
            raise ValueError('unsupported action or memory disabled')
        messages.append({'role': 'user', 'content': json.dumps(result, ensure_ascii=False)})
    raise ValueError('agent tool-turn limit without a submitted candidate')
