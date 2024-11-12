import plotly.graph_objects as go
import pandas as pd
from scipy.stats import describe
from scipy.stats._stats_py import DescribeResult

if __name__ == "__main__":
    # Define the path to the CSV file
    for dynamics_name in ["BSModel", "OUModel"]:
        csv_fpath = f'../../lecture4/Sterminals_{dynamics_name}.csv'  # Replace with your actual file path

        # Read the CSV file
        data = pd.read_csv(csv_fpath, header=None)

        # Extract the first column as values
        values: pd.Series = data.iloc[:, 0]
        stat: DescribeResult = describe(values, axis=0)

        # Prepare a nicely formatted statistics text
        stat_text = (
            f"nobs: {stat.nobs}<br>"
            f"min/max: {stat.minmax[0]:.2f}/{stat.minmax[1]:.2f}<br>"
            f"mean: {stat.mean:.2f}<br>"
            f"variance: {stat.variance:.2f}<br>"
            f"skewness: {stat.skewness:.2f}<br>"
            f"kurtosis: {stat.kurtosis:.2f}"
        )

        # Create the histogram using go.Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=values, name="Values"))

        # Add the statistics text as an annotation
        fig.add_annotation(
            x=0.95, y=0.95, xref="paper", yref="paper",
            text=stat_text,
            showarrow=False,
            align="left",
            font=dict(size=12),
            bordercolor="black", borderwidth=1, borderpad=4,
            bgcolor="white", opacity=0.8
        )

        # Update layout with a title
        fig.update_layout(
            title=f"Histogram of Values with Summary Statistics: {dynamics_name}",
            xaxis_title="Values",
            yaxis_title="Frequency",
        )

        # Display the plot
        fig.show()
