"""Convert LongMemEval dialogue inputs to traceable sources, without labels."""
from datetime import datetime


def time_metadata(raw):
    result = {'raw_session_time': raw, 'known_at': None,
              'time_precision': 'unknown', 'timezone': 'unknown',
              'known_at_basis': 'unknown'}
    if not isinstance(raw, str):
        return result
    try:
        parsed = datetime.strptime(raw, '%Y/%m/%d (%a) %H:%M')
    except ValueError:
        return result
    result.update(known_at=parsed.isoformat(timespec='minutes'),
                  time_precision='minute', known_at_basis='session_time')
    return result


def adapt_sessions(conversation_id, session_ids, session_dates, sessions):
    """Accept only dialogue fields: no question, answer, or gold labels."""
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError('conversation id required')
    if not all(isinstance(x, list) for x in (session_ids, session_dates, sessions)):
        raise ValueError('session fields must be lists')
    if not (len(session_ids) == len(session_dates) == len(sessions)):
        raise ValueError('session arrays differ in length')
    if any(not isinstance(sid, str) or not sid.strip() for sid in session_ids):
        raise ValueError('session id required')
    if len(set(session_ids)) != len(session_ids):
        raise ValueError('duplicate session id in conversation')
    sources = []
    for sid, date, turns in zip(session_ids, session_dates, sessions):
        if not isinstance(turns, list):
            raise ValueError('session turns must be a list')
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict) or not isinstance(turn.get('role'), str) or not turn['role'].strip() or not isinstance(turn.get('content'), str) or not turn['content'].strip():
                raise ValueError('turn role and content required')
            source = {'dataset_id': 'longmemeval_s_cleaned',
                      'conversation_id': conversation_id, 'session_id': sid,
                      'turn_id': str(index), 'speaker': turn['role'],
                      'text': turn['content']}
            source.update(time_metadata(date))
            sources.append(source)
    return sources
