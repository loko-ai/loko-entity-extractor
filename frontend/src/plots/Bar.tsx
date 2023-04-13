import { useColorModeValue } from '@chakra-ui/react';
import { BarDatum, BarSvgProps, ResponsiveBar } from '@nivo/bar';
import { getAxisTooltip } from './constants';

export function Bar({ data, ...rest }: BarProps) {
  const textColor = useColorModeValue('#222222', '#C9C9C9');
  const gridColor = useColorModeValue('#ddd', '#444');
  const k = Object.keys(data);
  const max = Math.max(...Object.values(data));
  
  const distro = k.map((kk) => {var obj = {};
                                  obj['id'] = kk;
                                  obj[kk] = data[kk];
                                  return obj});

    const theme = {
      fontSize: '14px',
      textColor: '#A9A9A9',
      grid: {
        line: {
          stroke: '#dddddd',
          strokeWidth: '.1px',

        }
      },
      axis: {
        domain: {
          line: {
            stroke: '#777777',
            strokeWidth: 0
          }
        },
        ticks: {
          text: {
            fontSize: '12px'
          }
        },
        legend: {
          text: {
            fontSize: '16px'
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
    <ResponsiveBar
      data={distro}
      keys={k}
      theme={theme}
      colors={['#800000']}
      borderRadius={"10"}
      margin={{
          top: 10,
          right: 10,
          bottom: 80,
          left: 150
        }}
      padding={.3}
      innerPadding={10}
      labelSkipWidth={max/10}
      //labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
      label={d => `${d.value}`}
      enableGridX={true}
      enableGridY={false}
      // minValue={.1}
      layout='horizontal'
      axisTop={null}
      axisBottom={{
            tickSize: 0,
            tickPadding: 5,
            tickRotation: -45,
            legendOffset: 80,
            legendPosition: 'middle',
            format: (v) => getAxisTooltip(v),
        }}
        axisLeft={{
            tickSize: 0,
            tickPadding: 25,
            tickRotation: 0,
            legendPosition: 'middle',
            legendOffset: -100,
            format: (v) => getAxisTooltip(v),
        }}
        
    />
  );
}
