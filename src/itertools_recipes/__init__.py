#automatically made via script
#helps mypy to correctly see the functions of the package

from .itertools_recipes import version as __version__

import typing as _typing
if not _typing.TYPE_CHECKING:

    from itertools import *
    try:
        from more_itertools import *
    except ImportError:
        pass
    from .shared_recipes import *
    from .itertools_recipes import *
else: 
  
    from itertools import (
    combinations_with_replacement,
    permutations,
    combinations,
    zip_longest,
    filterfalse,
    accumulate,
    #pairwise,
    takewhile,
    dropwhile,
    compress,
    product,
    starmap,
    groupby,
    islice,
    repeat,
    chain,
    cycle,
    count,
    tee
)

    try:
        from more_itertools import (
    #random_combination_with_replacement,
    distinct_combinations,
    distinct_permutations,
    UnequalIterablesError,
    duplicates_everseen,
    duplicates_justseen,
    #random_permutation,
    #random_combination,
    consecutive_groups,
    interleave_longest,
    substrings_indexes,
    interleave_evenly,
    groupby_transform,
    permutation_index,
    windowed_complete,
    combination_index,
    always_reversible,
    #unique_justseen,
    before_and_after,
    #unique_everseen,
    unique_in_window,
    nth_combination,
    always_iterable,
    #random_product,
    islice_extended,
    circular_shifts,
    nth_permutation,
    #sliding_window,
    unique_to_each,
    set_partitions,
    make_decorator,
    sort_together,
    zip_broadcast,
    callback_iter,
    product_index,
    filter_except,
    numeric_range,
    split_before,
    #iter_except,
    time_limited,
    chunked_even,
    SequenceView,
    split_after,
    nth_or_last,
    #dotproduct,
    #first_true,
    #roundrobin,
    repeat_last,
    AbortThread,
    side_effect,
    #repeatfunc,
    nth_product,
    count_cycle,
    repeat_each,
    value_chain,
    intersperse,
    run_length,
    split_when,
    triplewise,
    all_unique,
    partitions,
    zip_offset,
    substrings,
    map_reduce,
    distribute,
    interleave,
    #partition,
    strictly_n,
    difference,
    #all_equal,
    split_into,
    map_except,
    #is_sorted,
    #tabulate,
    #pairwise,
    #powerset,
    zip_equal,
    with_iter,
    #pad_none,
    #quantify,
    countable,
    subslices,
    exactly_n,
    mark_ends,
    #consume,
    split_at,
    peekable,
    #iterate,
    consumer,
    collapse,
    #ncycles,
    adjacent,
    #grouper,
    #chunked,
    #flatten,
    seekable,
    windowed,
    ichunked,
    convolve,
    padnone,
    stagger,
    prepend,
    recipes,
    rlocate,
    collate,
    replace,
    minmax,
    divide,
    bucket,
    #unzip,
    raise_,
    map_if,
    locate,
    rstrip,
    sliced,
    sample,
    lstrip,
    padded,
    #ilen,
    #tail,
    first,
    strip,
    #take,
    #nth,
    last,
    only,
    more,
    one,
    spy
)
    except ImportError:
        pass
        
    from .shared_recipes import (
    random_combination_with_replacement,
    random_combination,
    random_permutation,
    unique_justseen,
    unique_everseen,
    random_product,
    sliding_window,
    iter_except,
    first_true,
    roundrobin,
    is_sorted,
    groupwise,
    partition,
    pad_none,
    tabulate,
    pairwise,
    quantify,
    grouper,
    chunked,
    flatten,
    consume,
    ncycles,
    tail,
    ilen,
    nth
)
    from .itertools_recipes import (
    iteratefunc_with_index,
    interesting_lines,
    iteratefunc2ord,
    dual_accumulate,
    flatten_total,
    flatten_level,
    alternatesign,
    recursive_map,
    ida_y_vuelta,
    iteratefunc,
    repeatfunc,
    strip_last,
    dotproduct,
    vectorsum,
    alternate,
    lookahead,
    all_equal,
    rotations,
    range_of,
    dict_zip,
    powerset,
    splitAt,
    iterate,
    isplit,
    irange,
    rindex,
    ijoin,
    imean,
    unzip,
    take,
    skip
)

    __all__ = [
    'random_combination_with_replacement',
    'combinations_with_replacement',
    'iteratefunc_with_index',
    'distinct_combinations',
    'distinct_permutations',
    'UnequalIterablesError',
    'duplicates_everseen',
    'duplicates_justseen',
    'consecutive_groups',
    'interleave_longest',
    'substrings_indexes',
    'random_permutation',
    'random_combination',
    'interleave_evenly',
    'groupby_transform',
    'permutation_index',
    'windowed_complete',
    'interesting_lines',
    'combination_index',
    'always_reversible',
    'before_and_after',
    'unique_in_window',
    'unique_justseen',
    'nth_combination',
    'always_iterable',
    'unique_everseen',
    'islice_extended',
    'circular_shifts',
    'nth_permutation',
    'dual_accumulate',
    'iteratefunc2ord',
    'unique_to_each',
    'set_partitions',
    'random_product',
    'sliding_window',
    'make_decorator',
    'sort_together',
    'zip_broadcast',
    'callback_iter',
    'flatten_level',
    'product_index',
    'flatten_total',
    'filter_except',
    'numeric_range',
    'alternatesign',
    'recursive_map',
    'permutations',
    'combinations',
    'split_before',
    'ida_y_vuelta',
    'time_limited',
    'chunked_even',
    'SequenceView',
    'split_after',
    'iteratefunc',
    'nth_or_last',
    'filterfalse',
    'zip_longest',
    'repeat_last',
    'AbortThread',
    'side_effect',
    'iter_except',
    'nth_product',
    'count_cycle',
    'repeat_each',
    'value_chain',
    'intersperse',
    'run_length',
    'strip_last',
    'split_when',
    'triplewise',
    'all_unique',
    'dotproduct',
    'partitions',
    'zip_offset',
    'substrings',
    'map_reduce',
    'first_true',
    'roundrobin',
    'distribute',
    'interleave',
    'repeatfunc',
    'strictly_n',
    'difference',
    'accumulate',
    'split_into',
    'map_except',
    'vectorsum',
    'lookahead',
    'zip_equal',
    'partition',
    'rotations',
    'dropwhile',
    'with_iter',
    'alternate',
    'countable',
    'groupwise',
    'subslices',
    'all_equal',
    'takewhile',
    'exactly_n',
    'is_sorted',
    'mark_ends',
    'tabulate',
    'pairwise',
    'split_at',
    'powerset',
    'peekable',
    'consumer',
    'pad_none',
    'collapse',
    'quantify',
    'range_of',
    'dict_zip',
    'compress',
    'adjacent',
    'seekable',
    'windowed',
    'ichunked',
    'convolve',
    'padnone',
    'stagger',
    'product',
    'consume',
    'prepend',
    'groupby',
    'iterate',
    'recipes',
    'rlocate',
    'collate',
    'ncycles',
    'splitAt',
    'grouper',
    'chunked',
    'flatten',
    'replace',
    'starmap',
    'minmax',
    'divide',
    'irange',
    'repeat',
    'bucket',
    'raise_',
    'map_if',
    'locate',
    'rstrip',
    'rindex',
    'sliced',
    'sample',
    'isplit',
    'lstrip',
    'islice',
    'padded',
    'count',
    'chain',
    'ijoin',
    'cycle',
    'imean',
    'unzip',
    'first',
    'strip',
    'last',
    'only',
    'ilen',
    'tail',
    'more',
    'take',
    'skip',
    'nth',
    'one',
    'spy',
    'tee'
]


