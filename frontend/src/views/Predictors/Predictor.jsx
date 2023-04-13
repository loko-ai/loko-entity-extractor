import { Box, HStack, IconButton, Spacer, Stack, Tag } from "@chakra-ui/react";
import { RiDeleteBin4Line } from "react-icons/ri";

export function Predictor({ name, onDelete, ...rest }) {
  return (
    <HStack
      bg="gray.200"
      borderRadius={"10px"}
      w="100%"
      py="0.5rem"
      px="1rem"
      {...rest}
    >
      <Stack spacing={0}>
        <HStack color={"pink.500"}>
          <Box>{name}</Box>
          <Tag borderRadius={"10px"} p=".3rem" bg="pink.200" fontSize="xs">
            Not fitted
          </Tag>
        </HStack>
        <HStack fontSize={"xs"}>
          <Stack spacing="0">
            <Box>Model</Box>
            <Box>auto</Box>
          </Stack>
          <Stack spacing="0">
            <Box>transformer</Box>
            <Box>auto</Box>
          </Stack>
        </HStack>
      </Stack>
      <Spacer />
      <HStack>
        <IconButton
          size="sm"
          borderRadius={"full"}
          icon={<RiDeleteBin4Line />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onDelete(e);
          }}
        />
      </HStack>
    </HStack>
  );
}
