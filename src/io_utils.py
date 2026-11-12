"""Atomic result-file writing."""
import os,json,time,sys

def resilient_atomic_json(path,value):
    """Retry transient sharing locks; progress display cannot abort durable collection."""
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        for attempt in range(8):
            try:
                os.replace(temp,path)
                return True
            except PermissionError:
                if attempt==7:raise
                time.sleep(min(.05*2**attempt,1.))
    except PermissionError:
        if path.name!='progress.json':raise
        # Request start/end records were already appended and fsynced. The next
        # completed request will refresh the display; final execution summary is critical.
        print('Warning: progress display is temporarily locked; durable request journal retained.',file=sys.stderr,flush=True)
        return False
    finally:
        try:temp.unlink(missing_ok=True)
        except PermissionError:pass
