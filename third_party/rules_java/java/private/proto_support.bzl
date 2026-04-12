def compile(*, injecting_rule_kind, enable_jspecify, include_compilation_info, **kwargs):
    return java_common.compile(**kwargs)

def merge(providers, *, merge_java_outputs = True, merge_source_jars = True):
    return java_common.merge(providers)
