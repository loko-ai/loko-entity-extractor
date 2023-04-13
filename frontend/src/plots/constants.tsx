import { Tooltip } from '@chakra-ui/react';

const nivoTheme = {
  textColor: '#888',
  legends: {
    text: {
      color: '#888',
    },
  },
  tooltip: {
    container: {
      color: 'black',
    },
  },
  axis: {
    domain: {
      line: {
        stroke: '#888',
      },
    },
  },
  grid: {
    line: {
      stroke: '#888',
    },
  },
};

function getAxisTooltip(label: string) {
  return label.length > 10 ? (
    <Tooltip label={label}>
      <tspan>{`${label.substring(0, 10)}...`}</tspan>
    </Tooltip>
  ) : (
    label
  );
}

export { nivoTheme, getAxisTooltip };
