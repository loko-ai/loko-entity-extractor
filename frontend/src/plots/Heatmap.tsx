import { ResponsiveHeatMap } from '@nivo/heatmap';
import { getAxisTooltip } from './constants';


export function Heatmap({ data, labels, xLegend, yLegend, ...rest }: HeatmapProps) {

  const cm = [];
  data.forEach((row, i) => {
        const data_tmp = [];
        let tot = row.reduce((r, col) => r=r+col, 0);
        row.forEach((col, j) => {
            data_tmp[j] = {"x": labels[j], "y": (col>0) ? (col/tot) : (col), "value": col};
        });
        cm[i] = {"id": labels[i], "data": data_tmp};
        });
  
  console.log(cm);

  const theme = {
      fontSize: '12px',
      textColor: '#A9A9A9',
      axis: {
        ticks: {
          text: {
            fontSize: '12px'
          }
        },
        legend: {
          text: {
            fontSize: '14px'
          }
        },
      },
      tooltip: {
        container: {
          background: "#ffffff",
          color: "#333333",
          fontsize: "12px",
        }
      }
    };

  return (
        <ResponsiveHeatMap
        data={cm}
        forceSquare={true}
        theme={theme}
        borderRadius={"10"}
        borderWidth={"8"}
        borderColor="#222222"
        label={d => `${d.data.value}`}
        colorBy='id'
        margin={{
          top: 15,
          right: 10,
          bottom: 120,
          left: 5
        }}
        colors={{
            type: 'sequential',
            scheme: 'oranges'
        }}
        axisTop={null}
        axisBottom={{
            tickSize: 0,
            tickPadding: 5,
            tickRotation: -45,
            legend: 'PREDICTED',
            legendOffset: 80,
            legendPosition: 'middle',
            format: (v) => getAxisTooltip(v),
        }}
        axisLeft={{
            tickSize: 0,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'TRUE',
            legendPosition: 'middle',
            legendOffset: -100,
            format: (v) => getAxisTooltip(v),
        }}
        animate={true}
      />
  );
}
