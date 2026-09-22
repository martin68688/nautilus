import pathlib,json,urllib.request,time
p=pathlib.Path('/etc/autoresearch/apizh-terra-low.env');v=dict(x.split('=',1) for x in p.read_text().splitlines() if '=' in x)
assert v['OPENAI_BASE_URL']=='https://apizh-ai.com/v1'
body={'model':'gpt-5.6-terra','reasoning_effort':'low','messages':[{'role':'user','content':'Reply with only OK.'}],'max_tokens':128}
req=urllib.request.Request(v['OPENAI_BASE_URL']+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+v['OPENAI_API_KEY'],'Content-Type':'application/json'})
t=time.time();receipt={'endpoint':v['OPENAI_BASE_URL'],'requested_model':body['model'],'reasoning_effort':'low','started_unix':t,'scope':'relay transport real generation only; harness memory probe remains required'}
try:
 with urllib.request.urlopen(req,timeout=120) as res:
  d=json.load(res);receipt.update(http_status=res.status,response_model=d.get('model'),passed=bool(d.get('choices') and d['choices'][0].get('message',{}).get('content')),usage=d.get('usage'))
except Exception as e:receipt.update(passed=False,error_type=type(e).__name__,http_status=getattr(e,'code',None))
receipt['elapsed_seconds']=time.time()-t
pathlib.Path('/workspace/APIZH_TERRA_LOW_SMOKE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
