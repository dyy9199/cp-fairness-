def subgroup_confusion_tables(y_true, y_pred, subgroup, subgroup_name='group'):

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        subgroup_name: subgroup
    })
    tables = {}
    for g, sub in df.groupby(subgroup_name):
        if len(sub) == 0:
            continue
        cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0,1])
        tab = pd.DataFrame(cm, index=['True 0','True 1'], columns=['Pred 0','Pred 1']).astype(int)


        tab['Row Total'] = tab.sum(axis=1)

        col_totals = tab[['Pred 0','Pred 1']].sum(axis=0)
        grand_total = int(col_totals.sum())
        total_row = pd.DataFrame(
            [[int(col_totals['Pred 0']), int(col_totals['Pred 1']), grand_total]],
            index=['Col Total'],
            columns=['Pred 0','Pred 1','Row Total']
        )

        out = pd.concat([tab, total_row])
        tables[g] = out

    return tables

def pretty_print_tables(tables, title=''):
    if title:
        print(f"\n{title}")
    for g, t in tables.items():
        print(f"\n=== {g} ===")
        print(t)

